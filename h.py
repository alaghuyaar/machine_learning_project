from utils import log
import logging

# g = logging.getLogger(__name__)
# print(g)


# import logging


from pathlib import Path



try:
    path = Path('src/models/cleaning.py')
    with open(path,'r') as file:
        file.read(3)
except Exception as e:
    print(str(e))