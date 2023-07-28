from pyDataverse.models import Dataset
from pyDataverse.utils import read_file
from pyDataverse.api import NativeApi
from pyDataverse.models import Datafile
import tqdm
import os

if __name__ == "__main__":
    api = NativeApi('https://dataverse.harvard.edu', '778628b9-2a95-4e65-891e-1cbc8be0fe78')
    ds_pid = 'doi:10.7910/DVN/YNSDOU'

    df = Datafile()

    home_path = os.path.expanduser('~')

    oads_dir = f'{home_path}/projects/data/oads/oads_arw/ARW/'
    oads_images = [os.path.join(oads_dir, x) for x in os.listdir(oads_dir)]

    for image_path in tqdm.tqdm(oads_images, total=len(oads_images)):
        df.set({"pid": ds_pid, "filename": image_path})
        # df.get()
        resp = api.upload_datafile(ds_pid, image_path, df.json())
        resp.json()