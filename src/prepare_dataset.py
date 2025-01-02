import os
import json
import requests
from tqdm import tqdm
import logging
import concurrent.futures
import argparse
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_image(image_id: str, output_dir: str) -> bool:
    """Download a single COCO image by its ID."""
    # COCO image URL format
    url = f"http://images.cocodataset.org/train2017/{image_id}"
    output_path = os.path.join(output_dir, os.path.basename(image_id))
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        return True
    
    try:
        # Increased timeout and added retries for better reliability
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download {image_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download COCO images for instruction dataset')
    # Default workers to 80% of CPU cores for optimal performance while maintaining system responsiveness
    default_workers = int(multiprocessing.cpu_count() * 0.8)
    parser.add_argument('--instructions', type=str, default='data/instructions.json',
                      help='Path to instructions JSON file')
    parser.add_argument('--output-dir', type=str, default='data/images',
                      help='Output directory for downloaded images')
    parser.add_argument('--workers', type=int, default=default_workers,
                      help=f'Number of parallel workers (default: {default_workers}, based on CPU count)')
    args = parser.parse_args()

    logger.info(f"Using {args.workers} workers for parallel downloads")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load instruction data
    logger.info(f"Loading instructions from {args.instructions}")
    with open(args.instructions, 'r') as f:
        instructions = json.load(f)

    # Get unique image IDs
    image_files = list(set(item['image'] for item in instructions))
    total_images = len(image_files)
    logger.info(f"Found {total_images} unique images to download")

    # Download all images with a single progress bar
    success_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_image = {
            executor.submit(download_image, image_id, args.output_dir): image_id
            for image_id in image_files
        }

        with tqdm(total=total_images, desc="Downloading COCO images", unit='img') as pbar:
            for future in concurrent.futures.as_completed(future_to_image):
                image_id = future_to_image[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error downloading {image_id}: {str(e)}")
                    failed_count += 1
                pbar.update(1)
                pbar.set_postfix({'success': success_count, 'failed': failed_count}, refresh=True)

    logger.info(f"Download complete. Success: {success_count}, Failed: {failed_count}")
    if failed_count > 0:
        logger.warning(f"Failed to download {failed_count} images. Check the logs for details.")

if __name__ == "__main__":
    main() 