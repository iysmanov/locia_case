import importlib.util
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

PUBLIC_LINK = "https://disk.360.yandex.ru/d/cGxh0ny6NI7NVw"
RESOURCE_URL = "https://cloud-api.yandex.net/v1/disk/public/resources"
DOWNLOAD_URL = f"{RESOURCE_URL}/download"


class YandexDiskClient:
    def __init__(self, public_link: str, session: Optional[requests.Session] = None) -> None:
        self.public_link = public_link
        self.session = session or requests.Session()

    def _list_dir(self, path: str = "") -> List[dict]:
        # Рекурсивно обходит папки и собирает информацию о всех файлах
        items: List[dict] = []
        offset = 0
        while True:
            params = {"public_key": self.public_link, "path": path, "limit": 1000, "offset": offset}
            resp = self.session.get(RESOURCE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            embedded = data.get("_embedded", {})
            batch = embedded.get("items", [])
            if not batch:
                break
            for entry in batch:
                if entry.get("type") == "dir":
                    items.extend(self._list_dir(entry["path"]))
                elif entry.get("type") == "file":
                    items.append(entry)
            limit = embedded.get("limit", len(batch))
            total = embedded.get("total", offset + len(batch))
            offset += limit
            if offset >= total:
                break
        return items

    def list_files(self) -> List[dict]:
        return self._list_dir()

    def _download_url(self, path: str) -> str:
        resp = self.session.get(DOWNLOAD_URL, params={"public_key": self.public_link, "path": path}, timeout=30)
        resp.raise_for_status()
        href = resp.json().get("href")
        if not href:
            raise RuntimeError(f"Не удалось получить ссылку для скачивания: {path}")
        return href

    @staticmethod
    def _relative_path(yandex_path: str) -> Path:
        rel = yandex_path.split(":", 1)[-1].lstrip("/")
        return Path(rel)

    def download_file(self, file_info: dict, destination_root: Path) -> Path:
        # Качаем файл и сохраняем в ту же структуру папок
        rel_path = self._relative_path(file_info["path"])
        local_path = destination_root / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        href = self._download_url(file_info["path"])
        with self.session.get(href, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
        return local_path


def is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def get_base_dir() -> Path:
    # Определяем среду: Google Colab или локальный компьютер, и готовим папку вывода
    if is_colab():
        try:
            from google.colab import drive
        except ModuleNotFoundError as err:
            raise RuntimeError("Обнаружен Colab, но модуль google.colab недоступен.") from err
        drive.mount("/content/drive", force_remount=False)
        base = Path("/content/drive/MyDrive/yandex_disk_downloads")
    else:
        base = Path.cwd() / "yandex_disk_downloads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def read_tabular(path: Path) -> Optional[pd.DataFrame]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            return pd.read_excel(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suffix == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:
        print(f"Пропускаю {path} — ошибка чтения: {exc}")
    return None


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    # Приводим проблемные колонки к строкам и декодируем bytes, чтобы pyarrow не падал
    out = df.copy()
    obj_cols = out.columns

    def _to_text(val):
        if isinstance(val, bytes):
            try:
                return val.decode("utf-8")
            except UnicodeDecodeError:
                return val.decode("latin-1", errors="replace")
        return val

    for col in obj_cols:
        series = out[col]
        contains_text = series.apply(lambda v: isinstance(v, (str, bytes))).any()
        if contains_text or pd.api.types.is_object_dtype(series):
            out[col] = series.map(_to_text)
            # Приводим к строковому типу, сохраняя пропуски как <NA>
            out[col] = out[col].astype("string")
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Унифицируем названия колонок: убираем пробелы, неразрывные пробелы и приводим к нижнему регистру
    renames = {}
    for col in df.columns:
        name = "" if pd.isna(col) else str(col)
        normalized = name.replace("\xa0", " ").strip().lower()
        renames[col] = normalized
    return df.rename(columns=renames)


def align_columns_by_order(df: pd.DataFrame, base_columns: Optional[List[str]]) -> (pd.DataFrame, List[str]):
    """
    Выравнивает названия по порядку колонок. Если имя пустое, даём имя col_<index>.
    Если базовый порядок уже известен, применяем его по позициям.
    Возвращает (DataFrame, итоговый список колонок).
    """
    # нормализуем
    df = normalize_columns(df)
    col_names: List[str] = []
    for idx, col in enumerate(df.columns):
        name = "" if pd.isna(col) else str(col)
        name = name.replace("\xa0", " ").strip().lower()
        if not name:
            name = f"col_{idx}"
        col_names.append(name)

    if base_columns is None:
        # формируем базовый порядок по первому файлу
        rename_map = {old: new for old, new in zip(df.columns, col_names)}
        return df.rename(columns=rename_map), col_names

    # Если есть базовый порядок, выравниваем по позиции
    rename_map = {}
    for idx, old in enumerate(df.columns):
        target = base_columns[idx] if idx < len(base_columns) else f"col_{idx}"
        rename_map[old] = target
    aligned = df.rename(columns=rename_map)
    # если в этом файле меньше колонок, добавим пустые для недостающих позиций
    for idx in range(len(base_columns)):
        if base_columns[idx] not in aligned.columns:
            aligned[base_columns[idx]] = pd.NA
    # если базовый порядок короче, обрежем лишние столбцы
    aligned = aligned.reindex(columns=base_columns)
    return aligned, base_columns


def ask_files_already_downloaded() -> bool:
    # Узнаём у пользователя, нужно ли скачивать файлы заново
    answer = input("Файлы уже скачаны в целевую папку? (y/n): ").strip().lower()
    return answer in {"y", "yes", "д", "да"}


def collect_local_files(root: Path) -> List[Path]:
    # Собираем все файлы локально (рекурсивно), исключая итоговый parquet
    return [p for p in root.rglob("*") if p.is_file() and p.name != "combined.parquet"]


def build_parquet(downloaded_files: List[Path], destination_root: Path) -> Optional[Path]:
    tabular_suffixes = {".xlsx", ".xls", ".xlsm", ".csv", ".txt", ".tsv", ".parquet"}
    frames: List[pd.DataFrame] = []
    base_columns: Optional[List[str]] = None
    for file_path in downloaded_files:
        if file_path.suffix.lower() not in tabular_suffixes:
            continue
        df = read_tabular(file_path)
        if df is not None and not df.empty:
            df, base_columns = align_columns_by_order(df, base_columns)
            df = sanitize_for_parquet(df)
            df["__source_file"] = str(file_path.relative_to(destination_root))
            frames.append(df)
    if not frames:
        print("Табличные файлы не найдены, объединять нечего.")
        return None
    combined = pd.concat(frames, ignore_index=True)
    # Финальная нормализация типов перед записью
    combined = sanitize_for_parquet(combined)
    output_path = destination_root / "combined.parquet"
    try:
        combined.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Для записи в Parquet нужен пакет 'pyarrow' или 'fastparquet'. Установите один из них."
        ) from exc
    return output_path


def save_csv(combined: pd.DataFrame, destination_root: Path) -> Path:
    output_path = destination_root / "combined.csv"
    combined.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    base_dir = get_base_dir()
    print(f"Файлы сохраняются в: {base_dir}")
    if ask_files_already_downloaded():
        print("Скачивание пропущено по просьбе пользователя. Использую локальные файлы.")
        downloaded = collect_local_files(base_dir)
        if not downloaded:
            print("Локальные файлы не найдены. Попробуйте запустить заново и выбрать скачивание.")
            return
    else:
        client = YandexDiskClient(PUBLIC_LINK)
        print("Получаю список файлов на Яндекс Диске...")
        files = client.list_files()
        print(f"Найдено файлов: {len(files)}. Начинаю скачивание...")

        downloaded = []
        for idx, info in enumerate(files, start=1):
            try:
                local_file = client.download_file(info, base_dir)
                downloaded.append(local_file)
                print(f"[{idx}/{len(files)}] Скачан файл: {local_file}")
            except Exception as exc:
                print(f"[{idx}/{len(files)}] Не удалось скачать {info.get('name')}: {exc}")

    parquet_path = build_parquet(downloaded, base_dir)
    if parquet_path:
        print(f"Parquet файл создан: {parquet_path}")
        # Дополнительно сохраняем CSV для совместимости
        try:
            csv_path = save_csv(pd.read_parquet(parquet_path), base_dir)
            print(f"CSV файл создан: {csv_path}")
        except Exception as exc:
            print(f"Не удалось сохранить CSV: {exc}")
    else:
        print("Parquet не создан, так как табличные файлы не обработаны.")


if __name__ == "__main__":
    main()
