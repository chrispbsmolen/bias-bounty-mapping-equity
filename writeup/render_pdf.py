import sys, asyncio
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto("file://" + src); pg.wait_for_load_state("networkidle")
    pg.pdf(path=out, format="Letter", print_background=True, prefer_css_page_size=True,
           display_header_footer=True, header_template="<span></span>",
           footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#777;">Christopher Smolen (csmolen). Bias Bounty Mapping Equity Challenge. Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
           margin={"top":"14mm","bottom":"16mm","left":"17mm","right":"17mm"})
    b.close()
