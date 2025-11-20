import importlib.util
import os
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# Пороговые значения: чем больше файл/строк, тем легче выбираем библиотеку
SMALL_MAX_SIZE_MB = 50
SMALL_MAX_ROWS = 2_000_000
MEDIUM_MAX_SIZE_MB = 300


def is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def get_root_dir() -> Path:
    # Определяем корень поиска: в Colab попробуем примонтировать Google Drive
    if is_colab():
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            base = Path("/content/drive/MyDrive")
            if base.exists():
                return base
        except Exception as exc:
            print(f"Не удалось примонтировать Google Drive, работаю в /content: {exc}")
        return Path("/content")
    return Path.cwd()


def find_parquet_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def get_parquet_meta(path: Path) -> Tuple[Optional[int], Optional[int]]:
    # Пытаемся собрать метаданные без полного чтения
    if not is_available("pyarrow"):
        return None, int(path.stat().st_size)
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows
        size_bytes = pf.metadata.serialized_size
        if size_bytes is None or size_bytes <= 0:
            size_bytes = path.stat().st_size
        return rows, int(size_bytes)
    except Exception:
        return None, int(path.stat().st_size)


def choose_engine(rows: Optional[int], size_bytes: Optional[int]) -> str:
    size_mb = size_bytes / (1024 * 1024) if size_bytes is not None else None
    if rows is not None and size_mb is not None:
        if rows <= SMALL_MAX_ROWS and size_mb <= SMALL_MAX_SIZE_MB:
            return "ydata"
        if size_mb <= MEDIUM_MAX_SIZE_MB:
            return "skimpy"
        return "whylogs"
    if size_mb is not None:
        if size_mb <= SMALL_MAX_SIZE_MB:
            return "ydata"
        if size_mb <= MEDIUM_MAX_SIZE_MB:
            return "skimpy"
    return "whylogs"


def profile_with_ydata(df: pd.DataFrame, out_path: Path) -> None:
    if not is_available("ydata_profiling"):
        raise ImportError("ydata-profiling не установлен")
    from ydata_profiling import ProfileReport

    profile = ProfileReport(df, minimal=True, title="Parquet profile")
    profile.to_file(out_path)


def profile_with_skimpy(df: pd.DataFrame, out_path: Path) -> None:
    if not is_available("skimpy"):
        raise ImportError("skimpy не установлен")
    from skimpy import skim

    skim_df = skim(df)
    skim_df.to_csv(out_path, index=False)


def profile_with_whylogs(path: Path, out_path: Path) -> None:
    if not is_available("whylogs"):
        raise ImportError("whylogs не установлен")
    try:
        import pyarrow.parquet as pq
    except ImportError as err:
        raise ImportError("whylogs требует pyarrow для стриминга parquet") from err
    from whylogs import get_or_create_logger

    logger = get_or_create_logger(dataset_name=path.stem)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches():
        df_batch = batch.to_pandas()
        logger.log(df_batch)
    results = logger.close()
    # Сохраняем whylogs в бинарном формате профиля
    writer = results.writer("local")
    writer.write(file=out_path)


def load_dataframe(path: Path, limit_rows: Optional[int] = None) -> pd.DataFrame:
    if limit_rows:
        return pd.read_parquet(path, engine="pyarrow").head(limit_rows)
    return pd.read_parquet(path, engine="pyarrow")


def profile_file(path: Path, output_dir: Path) -> Optional[Path]:
    rows, size_bytes = get_parquet_meta(path)
    engine = choose_engine(rows, size_bytes)
    print(f"Файл: {path} | строк: {rows} | размер: {size_bytes} байт | движок: {engine}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if engine == "ydata":
        df = load_dataframe(path)
        out = output_dir / f"{path.stem}_ydata.html"
        profile_with_ydata(df, out)
        print(f"Профиль ydata сохранён в {out}")
        return out
    elif engine == "skimpy":
        df = load_dataframe(path, limit_rows=500_000)
        out = output_dir / f"{path.stem}_skimpy.csv"
        profile_with_skimpy(df, out)
        print(f"Профиль skimpy сохранён в {out}")
        return out
    else:
        out = output_dir / f"{path.stem}_whylogs.bin"
        profile_with_whylogs(path, out)
        print(f"Профиль whylogs сохранён в {out}")
        return out


def open_report(path: Path) -> None:
    # Открываем отчет в системной программе/браузере
    try:
        if is_colab():
            try:
                from IPython.display import HTML, display
                if path.suffix.lower() in {".html", ".htm"}:
                    display(HTML(path.read_text()))
                elif path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                    display(df.head(50))
                else:
                    print(f"Отчет сохранён: {path}")
            except Exception as exc:
                print(f"Не удалось отобразить отчет в Colab: {exc}")
            else:
                print(f"Отчет сохранён: {path}")
            return

        if path.suffix.lower() in {".html", ".htm", ".csv"}:
            webbrowser.open(path.as_uri())
            return
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", path], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        print(f"Не удалось автоматически открыть отчет {path}: {exc}")


def main() -> None:
    root = get_root_dir()
    parquet_files = find_parquet_files(root)
    if not parquet_files:
        print("Parquet файлы не найдены.")
        return
    output_dir = root / "parquet_profiles"
    generated: List[Path] = []
    for file_path in parquet_files:
        try:
            result = profile_file(file_path, output_dir)
            if result:
                generated.append(result)
        except ImportError as exc:
            print(f"Для файла {file_path} требуется установить пакет: {exc}")
        except Exception as exc:
            print(f"Не удалось построить профиль для {file_path}: {exc}")
    if generated:
        open_report(generated[0])
    else:
        print("Отчёты не сгенерированы.")


if __name__ == "__main__":
    sys.exit(main())
