import cv2 as cv
import numpy as np
import imutils
import pytesseract
from sudoku import Sudoku
import PIL
import time

#Pytesseract configuration
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
# luego= "-c tessedit_char_whitelist=0123456789"
myConfig = r"--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
# myConfig="outbase digits"

#Read video
video = cv.VideoCapture(0)
while(video.isOpened()):
    ret, frame = video.read()
    if(ret==True):
        #Convert to gray scale and apply a gaussian filter to avoid problems with colors and noise
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(img,(1,1),0)
        #We apply algorithm for finding contours
        thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
        thresh = cv.bitwise_not(thresh)
        contours = cv.findContours(thresh, cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)

        #Find sudoku grid
        sorted_contours=sorted(contours, key=cv.contourArea, reverse=True)

        sudoku_contour = []
        for contour in contours:
            polygon = cv.approxPolyDP(contour,15,True)
            if(len(polygon)) == 4:
               sudoku_contour=polygon
               break 

        if sudoku_contour == []:
            print("ERROR: CAN'T FIND SUDOKU CONTOUR")
        else:
            print("SUDOKU FOUND")
            #Perform a perspective transform so it takes up the whole window and image.
            contour_points = np.float32([sudoku_contour[0], sudoku_contour[3], sudoku_contour[1], sudoku_contour[2]])
            projection_points = np.float32([[0, 0],[450, 0],[0, 450],[450, 450] ])

            matrix = cv.getPerspectiveTransform(contour_points, projection_points)
            sudoku = cv.warpPerspective(blur, matrix, (450, 450))
            sudoku = cv.rotate(sudoku, cv.ROTATE_90_CLOCKWISE)
            cv.imshow("Sudoku Found",sudoku)
            cv.waitKey(1)

            sudoku_cells = []
            #Divide it in 81:
            columns = np.vsplit(sudoku,9)
            flag = True
            for x in columns:
                cells = np.hsplit(x,9)
                row_sudoku=[]
                count_zeros=0
                for cell in cells:
                    #apply a threshold to make it easier for pytesseract to detect the digits in the grid
                    cell_blur= cv.GaussianBlur(cell,(5,5),0)
                    ret,thresh = cv.threshold(cell_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                    #Pytesseract digit detection
                    text = pytesseract.image_to_string(thresh, config=myConfig)
                    # print(type(text))
                    # print(text)
                    if(text!=""):
                        text=int(text)
                    else:
                        text=0
                        count_zeros+=1
                    if(count_zeros>=9):
                        print("ERROR CONVERTING SUDOKU")
                        flag=False
                        break
                    row_sudoku.append(text)
                if(not flag):
                    break
                sudoku_cells.append(row_sudoku)

            if(flag):
                #Solve sudoku using the library
                puzzle= Sudoku(3,3,board=sudoku_cells)
                print(puzzle)
                solution=puzzle.solve()
                solution.show()
                
                #Print sudoku result over the original sudoku grid using the CV Put Text function
                if(solution.board!=0):
                    xStep = sudoku.shape[0] // 9 
                    yStep = sudoku.shape[1] // 9
                    fontScale=1
                    fontColor = (0,0,0)
                    thickness = 1
                    lineType = 2
                    xCoord = 20
                    yCoord = 40
                    for i in range(9):
                        for j in range(9):
                            if puzzle.board[i][j] == None:
                                cv.putText(sudoku,str(solution.board[i][j]),(xCoord, yCoord),cv.FONT_HERSHEY_SCRIPT_COMPLEX,fontScale,fontColor,thickness,lineType)
                            xCoord += xStep
                        xCoord = 20
                        yCoord += yStep
                    cv.destroyAllWindows()
                    cv.imshow("IMG SOLUCION",sudoku)
                    cv.waitKey(0)
                    time.sleep(20)

            #Show the sudoku with the result
            cv.imshow("Puzzle Outline", blur)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video.release()






