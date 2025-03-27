import pandas as pd
import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripAdvisorScraper:
    def __init__(self):
        self.setup_driver()
    
    def setup_driver(self):
        """Set up Chrome driver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in headless mode
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920x1080')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
    
    def _get_random_delay(self):
        """Add random delay between requests"""
        return random.uniform(2, 4)
    
    def _scroll_to_element(self, element):
        """Scroll element into view"""
        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
        time.sleep(1)
    
    def _extract_review_data(self, review_element):
        """Extract data from a single review element"""
        try:
            # Extract rating
            rating_elem = review_element.find_element(By.CSS_SELECTOR, 'span[class*="ui_bubble_rating bubble_"]')
            rating_class = rating_elem.get_attribute('class')
            rating = int(rating_class.split('_')[-1]) / 10
            
            # Extract review text
            review_text = review_element.find_element(By.CSS_SELECTOR, 'span[class*="QewHA"]').text.strip()
            
            # Extract date
            date_elem = review_element.find_element(By.CSS_SELECTOR, 'span[class*="euPKI"]')
            review_date = date_elem.text.strip()
            
            return {
                'rating': rating,
                'review_text': review_text,
                'date': review_date
            }
        except Exception as e:
            logger.error(f"Error extracting review data: {str(e)}")
            return None
    
    def scrape_destination_reviews(self, destination_urls, max_reviews_per_url=100):
        """Scrape reviews from multiple destination URLs"""
        all_reviews = []
        
        try:
            for url in destination_urls:
                logger.info(f"Scraping reviews from: {url}")
                self.driver.get(url)
                time.sleep(self._get_random_delay())
                
                # Wait for reviews to load
                try:
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[class*="review-container"]')))
                except TimeoutException:
                    logger.error(f"Timeout waiting for reviews to load on {url}")
                    continue
                
                reviews_scraped = 0
                last_height = self.driver.execute_script("return document.body.scrollHeight")
                
                while reviews_scraped < max_reviews_per_url:
                    # Find all review containers
                    review_elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[class*="review-container"]')
                    
                    for review_elem in review_elements[reviews_scraped:]:
                        if reviews_scraped >= max_reviews_per_url:
                            break
                            
                        self._scroll_to_element(review_elem)
                        review_data = self._extract_review_data(review_elem)
                        
                        if review_data:
                            all_reviews.append(review_data)
                            reviews_scraped += 1
                            logger.info(f"Scraped {reviews_scraped} reviews from current URL")
                    
                    # Scroll down to load more reviews
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)
                    
                    new_height = self.driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                
                time.sleep(self._get_random_delay())
                
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
        
        finally:
            self.driver.quit()
        
        reviews_df = pd.DataFrame(all_reviews)
        logger.info(f"Total reviews scraped: {len(reviews_df)}")
        return reviews_df
    
    def save_reviews_to_csv(self, reviews_df, filename='tripadvisor_reviews.csv'):
        """Save scraped reviews to CSV file"""
        try:
            reviews_df.to_csv(filename, index=False)
            logger.info(f"Successfully saved {len(reviews_df)} reviews to {filename}")
        except Exception as e:
            logger.error(f"Error saving reviews to CSV: {str(e)}")

def main():
    # Example destination URLs (popular tourist destinations)
    destination_urls = [
        'https://www.tripadvisor.com/Hotel_Review-g147293-d149213-Reviews-Excellence_Punta_Cana-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',
        'https://www.tripadvisor.com/Hotel_Review-g147293-d578371-Reviews-Secrets_Royal_Beach_Punta_Cana-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html',
        'https://www.tripadvisor.com/Hotel_Review-g147293-d218492-Reviews-Barcelo_Bavaro_Beach_Adults_Only-Punta_Cana_La_Altagracia_Province_Dominican_Republic.html'
    ]
    
    scraper = TripAdvisorScraper()
    reviews_df = scraper.scrape_destination_reviews(destination_urls, max_reviews_per_url=50)
    scraper.save_reviews_to_csv(reviews_df)

if __name__ == "__main__":
    main()
