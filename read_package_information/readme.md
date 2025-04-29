# 🧾 Automatizované zpracování etiket a OCR validace

Tento projekt slouží k automatizovanému zpracování etiket výrobků pomocí počítačového vidění (OpenCV) a OCR (Tesseract). Cílem je najít oblasti zájmu na etiketě, přečíst z nich text a porovnat ho s referenčními daty získanými z Excelu.

---

## 📁 Struktura projektu

```
📦 Projekt
├── main.py                         # Hlavní skript - řídí načítání dat, OCR a porovnání
├── utils_files.py                  # Pomocné funkce pro práci s JSON a Excel soubory
├── utils_image_processing.py       # Zpracování obrazu, OCR, výpočet podobnosti textu
```

---

## 🧠 Funkce

### ✅ `main.py`
- Spouští celý proces zpracování snímku etikety.
- Načítá JSON a Excel s referenčními daty.
- Z obrázku najde největší obdélník (etiketu), vypočítá měřítko a ořízne oblast.
- Vyhledá textové pozice a provede OCR s následným porovnáním s referenčními texty.
- Možnost detekce piktogramů nebo jiných oblastí.

### 🛠️ `utils_files.py`
- `load_json` / `save_json`: Načtení a uložení JSON souborů.
- `load_excel_to_dict`: Převod Excelu na list slovníků.
- `fill_json_from_excel`: Naplní JSON hodnotami z Excelu podle REF kódu.
- `find_row_by_ref`: Vyhledá konkrétní řádek podle REF kódu.

### 🖼️ `utils_image_processing.py`
- `get_biggest_polygon`: Najde největší polygon (etiketu) a vypočítá její úhel.
- `extract_text_from_frame`: Provede OCR nad vybranou oblastí obrázku.
- `text_similarity`: Porovná načtený text s referenčním.
- `detect_first_black_pixel`: Najde první textový bod v obrázku.
- `calculate_pixel_size`: Spočítá velikost pixelu na základě vzdálenosti kamery.
- `draw_rotated_rect`: Vykreslí textové boxy na obrázek.

---

### Instalace závislostí:
```bash
pip install opencv-python pytesseract pandas numpy
```

Nezapomeň mít nainstalovaný **Tesseract OCR** a dostupný v PATH. Např. pro Windows:
```bash
choco install tesseract
```

---

## 📸 Práce s obrázkem

- Projekt může buď:
  - 📁 Načíst statický obrázek (`TraumastemTafLight.png`),
  - 🎥 Nebo pracovat s živým obrazem z kamery (přes `camera_thread_HKVision`).

Nastavení v `main.py`:
```python
settings = {
    "use_camera": False,                # Přepnout na True pro použití kamery
    "text_compare": True,              # Provádět porovnání OCR vs. reference
    "snap_pictograms": True,           # Ukládat piktogramy
    "draw_pictograms_on_origin": True  # Vykreslit boxy do originálního obrázku
}
```

---

## 📈 Ukázkový průběh

1. Načte se obrázek etikety.
2. Najde se hlavní oblast (etiketa).
3. Spočítá se měřítko (px → mm).
4. Ořízne se oblast a načte se REF kód.
5. Načte se odpovídající řádek z Excelu.
6. Provádí se OCR pro jednotlivé textové pole a výstup se porovnává s referenční hodnotou.

---

📌 **Poznámka:** Všechny cesty k souborům (`.json`, `.xlsx`, obrázek) jsou v `main.py` zapsány natvrdo. V případě potřeby uprav dle své struktury projektu nebo použij CLI argumenty.
