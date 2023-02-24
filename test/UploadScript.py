import sys, os, imagehash, rawpy, imageio
from PIL import Image

if len(sys.argv)<2:
    print("Specify file path to source directory")
    sys.exit()

pathARW = sys.argv[1]   # set path to the source directory with ARW files [100MSDCF]
parentdir = os.path.abspath(os.path.join(pathARW, os.pardir)) # create JPG directory in same directory as 100MSDCF
pathJPG = parentdir+'/JPG'
newFolder = parentdir+'/'+"Supervisely" # to save resized JPG's in seperate folder 

if os.path.isdir(pathJPG):
    print("JPG folder already exists in this directory. Remove previous JPG folders to avoid duplicate uploads.")
    sys.exit()
if os.path.isdir(newFolder):
    print("Supervisely folder already exists in this directory. Remove previous Supervisely folders to avoid duplicate uploads.")
    sys.exit()

os.mkdir(pathJPG)
os.mkdir(newFolder)
dirARW = os.listdir(pathARW)

### convert to JPG
for i in dirARW:
    f, e = os.path.splitext(i)
    if e == ".ARW":
        with rawpy.imread(pathARW+'/'+i) as raw:
            rgb = raw.postprocess()
            imagename = f+".jpg"
            imageio.imsave(pathJPG+'/'+imagename, rgb) # save in JPG folder

### change filenames
# do this after converting as hashing doesn't work with ARW files
# dirJPG = os.listdir(pathJPG)

# jpgFiles = set() # create sets of photos same extensions
# arwFiles = set() 

# for item in dirJPG:
#     f, e = os.path.splitext(item)
#     jpgFiles.add(f)

# for item in dirARW:
#     f,e = os.path.splitext(item)
#     arwFiles.add(f) 
        
# for name in jpgFiles.intersection(arwFiles): # for common files
#     with Image.open(pathJPG+'/'+name+'.JPG') as im: # open then hash image
#         imHash = str(imagehash.dhash(im)) # use of difference hash because less chance of duplicate names
#     os.rename(os.path.join(pathJPG,name+'.JPG'),os.path.join(pathJPG,imHash+'.JPG')) # Rename jpg file
#     os.rename(os.path.join(pathARW,name+'.ARW'),os.path.join(pathARW,imHash+'.ARW')) # Rename json file

### change size of JPG's
for item in os.listdir(pathJPG):
    f, e = os.path.splitext(item) # extract filename & extension
    with Image.open(pathJPG+'/'+item) as im:
        w, h = im.size # get width, heigth
        new_w = int(w/4) # resize by factor 4
        new_h = int(h/4)
        imResize = im.resize((new_w,new_h), Image.Resampling.LANCZOS) 
        imName = newFolder+'/'+f+'.JPG'
        imResize.save(imName, 'JPEG') # save in the seperate Supervisely folder

