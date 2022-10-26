# Python 3 code to rename multiple
# files in a directory or folder
# WORKED
# importing os module
import os
os.chdir(r'E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\data2\output\train')

# Function to rename multiple files

def main():
    
    
    #'1','2'
    lst3=['0','1','2','G','H','I','J','R','Z']
    #lst=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    lst=['A','B','C','D','E','F','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y']
    for folder in lst3:
        for count, filename in enumerate(os.listdir(folder)):

                dst = f"{str(count)}.jpg"     # jst bracket bahar x laga do or hata do
                src =f"{folder}/{filename}" # foldername/filename, if .py file is outside folder
                dst =f"{folder}/{dst}"
                
                # rename() function will
                # rename all the files
                os.rename(src, dst)

# Driver Code
if __name__ == '__main__':

    # Calling main() function
    main()
