import exif

def readExif(fname):
   im_exif = exif.Image(fname)
   print("Has exif: ", im_exif.has_exif)
   if im_exif.has_exif:
      for i in dir(im_exif):
         if i != '_segments':
            print(f"{i} : {im_exif.get(i)}")