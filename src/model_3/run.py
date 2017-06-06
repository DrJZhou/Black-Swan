__author__ = 'guoyang3'

import os
def main():
    os.system("python generate_20min_volume.py")
    os.system("python preprocessing0.py")
    os.system("python preprocessing1.py")
    os.system("python preprocessing2.py")
    os.system("python preprocessing3.py")
    os.system("python preprocessing4.py")

    os.system("python model1.py")
    os.system("python model2.py")
    os.system("python model3.py")
    os.system("python model4.py")
    os.system("python model5.py")

    os.system("python stacking.py")

if __name__ == '__main__':
    main()
