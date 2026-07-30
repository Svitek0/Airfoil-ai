from playwright.sync_api import sync_playwright

# The app to keep awake.
APP_URL = "https://airfoilai.streamlit.app/"

# Text on Streamlit's wake button (matched loosely, case-insensitive).
WAKE_BUTTON_TEXT = "get this app back up"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        print(f"Opening {APP_URL}")
        # Streamlit issues an auth redirect (303) and then loads a JS app,
        # so wait for the network to settle rather than just the first response.
        page.goto(APP_URL, wait_until="networkidle", timeout=60_000)

        # Give the page a moment to render the sleeping screen if present.
        page.wait_for_timeout(4_000)

        # Look for the wake button. If it's there, the app was asleep.
        button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)

        if button.count() > 0:
            print("App was asleep - clicking the wake button.")
            button.first.click()
            # Wait for the app to actually boot after the click.
            page.wait_for_timeout(30_000)
            print("Wake button clicked; app should be starting.")
        else:
            print("No wake button found - app is already awake. Timer reset.")

        browser.close()
        print("Done.")


if __name__ == "__main__":
    main()
