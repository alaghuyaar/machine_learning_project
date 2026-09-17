import logging
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

def config_log():
    log_path = os.getenv('LOG_PATH')
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )