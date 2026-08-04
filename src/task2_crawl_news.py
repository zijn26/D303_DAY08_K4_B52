"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    # Đổi phương thức thanh toán
    "https://help.shopee.vn/portal/4/article/79128-H%C6%B0%E1%BB%9Bng-D%E1%BA%ABn-%C4%90%E1%BB%95i-Ph%C6%B0%C6%A1ng-Th%E1%BB%A9c-Thanh-To%C3%A1n-Cho-%C4%90%C6%A1n-H%C3%A0ng-Tr%E1%BA%A3-Tr%C6%B0%E1%BB%9Bc",

    # Thời gian và cách kiểm tra tiền hoàn
    "https://help.shopee.vn/portal/4/article/189473-%5BTr%E1%BA%A3-h%C3%A0ng/-Ho%C3%A0n-ti%E1%BB%81n%5D-Th%E1%BB%9Di-gian-nh%E1%BA%ADn-ti%E1%BB%81n-ho%C3%A0n-v%C3%A0-c%C3%A1ch-ki%E1%BB%83m-tra-ti%E1%BB%81n-ho%C3%A0n",

    # Đơn hàng sai trạng thái/chưa nhận được hàng
    "https://help.shopee.vn/portal/4/article/79084",

    # Các phương thức gửi hàng hoàn trả
    "https://help.shopee.vn/portal/4/article/189477-%5BTr%E1%BA%A3-h%C3%A0ng/-Ho%C3%A0n-ti%E1%BB%81n%5D-C%C3%A1c-ph%C6%B0%C6%A1ng-th%E1%BB%A9c-g%E1%BB%ADi-h%C3%A0ng-ho%C3%A0n-tr%E1%BA%A3-v%C3%A0-ph%C3%AD-ho%C3%A0n-tr%E1%BA%A3",

    # Lỗi thanh toán
    "https://help.shopee.vn/portal/4/article/84824",
]


async def crawl_article(url: str) -> dict:
    """Crawl một bài viết và trả về nội dung cùng metadata."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        if not result.success:
            raise RuntimeError(
                f"Không crawl được {url}: {result.error_message}"
            )

        metadata = result.metadata or {}
        markdown = result.markdown

        # Tương thích các phiên bản Crawl4AI khác nhau.
        if hasattr(markdown, "raw_markdown"):
            markdown = markdown.raw_markdown

        markdown = str(markdown or "").strip()

        if len(markdown) < 300:
            raise RuntimeError(
                f"Nội dung crawl từ {url} quá ngắn ({len(markdown)} ký tự)"
            )

        return {
            "url": url,
            "title": metadata.get("title", "Không xác định"),
            "date_crawled": datetime.now().astimezone().isoformat(),
            "content_markdown": markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())