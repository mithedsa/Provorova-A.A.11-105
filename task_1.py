import os
import requests
from bs4 import BeautifulSoup
from time import sleep


def download_pages(base_url, start_id, end_id, output_dir="downloaded_pages", index_file="index.txt"):
    os.makedirs(output_dir, exist_ok=True)
    index_entries = []

    for i in range(start_id, end_id + 1):
        url = f"{base_url}{i}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("h1", itemprop="name")
            description = soup.find("span", class_="readmore-container")

            book_info = []
            book_info.append(title.get_text(strip=True) if title else "No Title")
            book_info.append(description.get_text(strip=True) if description else "No Description")

            about_section = soup.find(lambda tag: tag.name == "h2" and "About this eBook" in tag.get_text())
            if about_section:
                table = about_section.find_next("table", class_="bibrec")
                if table:
                    for row in table.find_all("tr"):
                        header = row.find("th")
                        data = row.find("td")
                        if header and data:
                            book_info.append(f"{header.get_text(strip=True)}: {data.get_text(' ', strip=True)}")

            text_content = "\n\n".join(book_info)

            file_name = f"page_{i}.txt"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            index_entries.append(f"{i} {url}\n")
            print(f"Скачано: {url}")

        except requests.RequestException as e:
            print(f"Ошибка при скачивании {url}: {e}")

        sleep(1)

    with open(index_file, "w", encoding="utf-8") as index_f:
        index_f.writelines(index_entries)

    print("Готово!")


base_url = "https://www.gutenberg.org/ebooks/"
start_id = 1
end_id = 100
download_pages(base_url, start_id, end_id)
