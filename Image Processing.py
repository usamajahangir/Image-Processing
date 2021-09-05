####################          Signals and System       ######################
#############                  Semister-Project             #################

########################   Group-Members   ##################################

####     NAME     ################   CMD ID  ######## Syndicate
#   Usama Jahangir                   295503            MTS-A
#   Taimoor Asif
#   Muhammad Ameer Usman

##########################################   CODES   ##############################################
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from PIL import Image, ImageEnhance
############################################ Menu  ################################################
choice = 0
while choice != 8:
    choice = 0
    print("What do you want to perform ?\n"
          "1- Make a real time histogram (q to exit from this)\n"
          "2- Histogram of a picture\n"
          "3- Change Image Resolution\n"
          "4- Image Flip\n"
          "5- Image (Add && Subtract))\n"
          "6- Color Effects\n"
          "7- Image Concatenation\n"  
          "8- Exit\n")

    choice = int(input("Your Choice: "))

                        ###############   Real Time Histogram   #######################
    ############################################################################
    # This code is picked from a source which is cited in the Project report   #
    #                                                                          #
    #   Description: Uses OpenCV to display video from a camera or file        #
    #        and matplotlib to display and update either a grayscale or        #
    #        RGB histogram of the video in real time. For usage, type:         #
    #        > python real_time_histogram.py -h                                #
    #    Author: Najam Syed (github.com/nrsyed)                                #
    #    Created: 2018-Feb-07                                                  #
    ############################################################################

    if choice == 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--file',
                            help='Path to video file (if not using camera)')
        print ("What type of histogram you want\n"
               "1-        Gray-Scale\n"
               "Any Key-  RGB\n")
        schoice = int(input())
        if schoice == 1:
            parser.add_argument('-c', '--color', type=str, default='gray',
                                help='Color space: "gray" (default), "rgb", or "lab"')
        elif schoice != 1:
            parser.add_argument('-c', '--color', type=str, default='rgb',
               help='Color space: "gray" (default), "rgb", or "lab"')

        parser.add_argument('-b', '--bins', type=int, default=16,
            help='Number of bins per channel (default 16)')
        parser.add_argument('-w', '--width', type=int, default=0,
            help='Resize video to specified width in pixels (maintains aspect)')
        args = vars(parser.parse_args())


        # Configure VideoCapture class instance for using camera or file input.
        if not args.get('file', False):
            schoice = int(input("Which Camera you want to use:\n"
                           "1-         Other (Alert !! Only choose if there are atleast one camera expect built-in)\n"
                           "Any Key-   Built-in of your device\n"))
            if schoice == 1:
                capture = cv2.VideoCapture(1)
            else:
                capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(args['file'])

        color = args['color']
        bins = args['bins']
        resizeWidth = args['width']

        # Initialize plot.
        fig, ax = plt.subplots()
        if color == 'rgb':
            ax.set_title('Histogram (RGB)')
        elif color == 'lab':
            ax.set_title('Histogram (L*a*b*)')
        else:
            ax.set_title('Histogram (grayscale)')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Frequency')

        # Initialize plot line object(s). Turn on interactive plotting and show plot.
        lw = 3
        alpha = 0.5
        if color == 'rgb':
            lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
            lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
            lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')
        elif color == 'lab':
            lineL, = ax.plot(np.arange(bins), np.zeros((bins,)), c='k', lw=lw, alpha=alpha, label='L*')
            lineA, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='a*')
            lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='y', lw=lw, alpha=alpha, label='b*')
        else:
            lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
        ax.set_xlim(0, bins-1)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.ion()
        plt.show()

        # Grab, process, and display video frames. Update plot line object(s).
        while True:
            (grabbed, frame) = capture.read()

            if not grabbed:
                break

            # Resize frame to width, if specified.
            if resizeWidth > 0:
                (height, width) = frame.shape[:2]
                resizeHeight = int(float(resizeWidth / width) * height)
                frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                    interpolation=cv2.INTER_AREA)

            # Normalize histograms based on number of pixels per frame.
            numPixels = np.prod(frame.shape[:2])
            if color == 'rgb':
                cv2.imshow('RGB', frame)
                (b, g, r) = cv2.split(frame)
                histogramR = cv2.calcHist([r],[0], None, [bins], [0, 255]) / numPixels
                histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
                histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
                lineR.set_ydata(histogramR)
                lineG.set_ydata(histogramG)
                lineB.set_ydata(histogramB)
            elif color == 'lab':
                cv2.imshow('L*a*b*', frame)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                (l, a, b) = cv2.split(lab)
                histogramL = cv2.calcHist([l], [0], None, [bins], [0, 255]) / numPixels
                histogramA = cv2.calcHist([a], [0], None, [bins], [0, 255]) / numPixels
                histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
                lineL.set_ydata(histogramL)
                lineA.set_ydata(histogramA)
                lineB.set_ydata(histogramB)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Grayscale', gray)
                histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
                lineGray.set_ydata(histogram)
            fig.canvas.draw()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

                   ########################   Image Histogram  ########################

    if choice == 2:
        choice = input ("What you want : \n"
                        "1-       Give a path to image (Must be correct)\n"
                        "Any Key- Run for default image\n")
        if choice == 1:
            filepath = input ("Image Path : ")
            img = cv2.imread (filepath, 0)
        else:
            img = cv2.imread('2.PNG', 0)
        cv2.resize(img, (700, 700))
        plt.hist(img)
        plt.show()

                    ########################   Image Resolution  ########################

    if choice == 3:
        def resolve(img, bits):
            bits = bits % 9
            mtpr = pow(2, 8-bits)
            size = np.shape(img)
            new = np.zeros(size, dtype=np.uint8)
            new[::, ::] = ( ( ( img[::, ::] ) // mtpr ) ) * mtpr
            #dividing with mtpr becomes integer before it is multiplied with mtpr so its rounded off
            return new
        img = cv2.imread('2.PNG', 0)
        rimg = cv2.resize(img, (700,700))
        bit = int(input("Enter resolution bits :  "))
        new_img = resolve(rimg, bit)
        cv2.imshow("Grayscale-Image", rimg)
        cv2.imshow("Resolved-Image", new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

                   ##########################      Flipping      #########################

    if choice == 4:
        nimg = cv2.imread('2.PNG', 1)
        img = cv2.resize(nimg, (1080, 700))
        choice = int(input("\n1- Half Flipped\n"
                           "2- Full Flipped (Inverted)\n"))
        if choice == 1:
            row = np.shape(img)[0]
            col = np.shape(img)[1]
       #     uphalf = img[0: int(row/2)][0: col]
        #    lowerhalf = uphalf[::-1][0:]

            lowerhalf = img[int(row / 2):][0: col]
            uphalf = lowerhalf[::-1][0:]

            new = np.concatenate((uphalf, lowerhalf), axis=0)
            cv2.imshow("Flipped img", new)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            new_img = img[::-1][0:]
            cv2.imshow("Whole Flipped", new_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
                   ########################## Image Addition and Subtraction  #########################

    if choice == 5:
        img_1 = cv2.imread('2.PNG')
        img_1_fix = cv2.resize(img_1, (700,700))
        img_2 = cv2.imread('2.PNG')
        img_2_fix = cv2.resize(img_2, (700,700))
        choice = int(input("\n1- Add images\n"
                           "2- Subtract images\n"))
        if choice == 1:
            combined_img = cv2.addWeighted(img_1_fix, 0.5, img_2_fix, 0.1, 0)
            cv2.imshow('Image', combined_img)
            check = cv2.waitKey(0) & 0xFF
            if check == 27:
                cv2.destroyAllWindows()
        else:
            delete_img = cv2.subtract(img_1_fix, img_2_fix)
            cv2.imshow('Image', delete_img)
            check = cv2.waitKey(0) & 0xFF

            if check == 27:
                cv2.destroyAllWindows()
                   ##########################     Colors          #########################

    if choice == 6:
        factor = float(input("Enter factor for Enhancement (grayscale = 0 & Original = 1) :"))
        im = Image.open(r"2.PNG")
        im3 = ImageEnhance.Color(im)
        im3.enhance(factor).show()

        ##########################     Image Concatenate          #########################

    if choice == 7:
        n = cv2.imread("2.PNG", 1)
        n2 = cv2.imread("color.png", 1)
        img2 = cv2.resize(n2, (1080,700))
        img = cv2.resize(n, (1080, 700))
        row = np.shape(img)[0]
        col = np.shape(img)[1]
        uphalf = img[0: int(row / 2)][0: col]
        lowerhalf = uphalf[::-1][0:]

        row = np.shape(img2)[0]
        col = np.shape(img2)[1]
        uphalf2 = img2[0: int(row / 2)][0: col]
        lowerhalf2 = uphalf2[::-1][0:]

        new = np.concatenate((uphalf, lowerhalf2), axis=0)
        cv2.imshow("Concatenated Image", new)
        cv2.waitKey(0)
        cv2.destroyAllWindows()