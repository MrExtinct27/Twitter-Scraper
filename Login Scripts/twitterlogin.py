from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # visible browser
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://twitter.com/login")
    
    print("Please log in manually in the browser window...")
    print("Once logged in, close the browser to save the session.")

    page.wait_for_timeout(60000)  # Wait 60 seconds for manual login
    context.storage_state(path="twitter_auth.json")
    browser.close()
