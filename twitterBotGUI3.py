import tkinter as tk
from tkinter import font

import time
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, HORIZONTAL
from tkinter.ttk import Progressbar
import threading
from tkinter.messagebox import showinfo

from combined_utils_4 import *
from summarizor import * 

##################################################################################
### Logging in and out to twitter handling !!

def popup_bonus():
    win = tk.Toplevel()
    win.wm_title("Login Window")
    win.minsize(200, 200)
    
    l = tk.Label(win, text="Log in to Twitter", font=('Helvetica', 16, 'bold'))
    l.grid(row=0, column=1, columnspan=9, ipadx=5, ipady=5, padx=20, pady=10)

    b = ttk.Button(win, text="Okay", command=win.destroy)
    b.grid(row=4, column=1, columnspan=9, ipadx=5, ipady=5, padx=20, pady=10)
    
    user = makeentry(win, "User name:", 10)
    password = makeentry2(win, "Password:", 10, show="*")

def makeentry(win, caption, width=10, **options):
    Label(win, text= "User name:", font=('Helvetica', 12, 'bold')).grid(column=1, columnspan=2, row=1, ipadx=5, ipady=5, padx=20, pady=10)
    entry = Entry(win, **options)
    entry.config(width=width)
    entry.grid(column=3, columnspan=8, row=1, ipadx=50, ipady=5, padx=20, pady=10)
    return entry    

def makeentry2(win, caption, width=10, **options):
    Label(win, text="Password:", font=('Helvetica', 12, 'bold')).grid(column=1, columnspan=2, row=2, ipadx=5, ipady=5, padx=20, pady=10)
    entry = Entry(win, **options)
    entry.config(width=width)
    entry.grid(column=3, columnspan=8, row=2, ipadx=50, ipady=5, padx=20, pady=10)
    return entry


#def popup_showinfo_loggout():
#    showinfo("Window", "You habe been succesfully logged out!")

##################################################################################
### Displaying Article !!

def displayArticle(article):
    win = tk.Toplevel()
    win.wm_title("Whole Article")
    win.minsize(400, 400)
    
    
    text = Text(win)
    text.insert(INSERT, article)
    text.grid(column=0, columnspan=9, row=1, ipadx=20, ipady=10, padx=20, pady=10)


    b = ttk.Button(win, text="Close", command=win.destroy)
    b.grid(row=4, column=0, columnspan=9, ipadx=5, ipady=5, padx=20, pady=10)
    
##################################################################################
### Displaying Sumary !!

def displaySummary(summary):
    win = tk.Toplevel()
    win.wm_title("Summary of the article")
    win.minsize(400, 100)
    
    
    text = Text(win)
    text.insert(INSERT, summary)
    text.grid(column=0, columnspan=9, row=1, ipadx=20, ipady=10, padx=20, pady=10)

    b = ttk.Button(win, text="Close", command=win.destroy)
    b.grid(row=4, column=0, columnspan=9, ipadx=5, ipady=5, padx=20, pady=10)
    
##################################################################################
### Progresbar for while creating summary !!


def summarize_article(url):
    win = tk.Toplevel()
    win.wm_title("Summarizing ... ")
    win.minsize(400, 300)
    win.configure(background = '#FFFFFF')

    def progress(currentValue):
        progressbar["value"]=currentValue
     
    hedder = Label(win, background= '#1da1f2', font=('Helvetica', 30, 'bold'), text="Summarizer is working ... ", foreground='#FFFFFF')
    hedder.grid(column = 1, columnspan=8, row = 2, ipadx = 35, pady = 0, sticky=N+S+E+W)
    
    maxValue=100 
    currentValue=0
    progressbar=ttk.Progressbar(win,orient="horizontal",length=300,mode="determinate")
    progressbar.grid(row=4, column=1, columnspan=9, ipadx=5, ipady=15, padx=20, pady=30)
    progressbar["value"]=currentValue
    progressbar["maximum"]=maxValue
    
    summary = summarizor(url)
    print(summary)
    
    divisions=20
    for i in range(divisions):
        currentValue=currentValue+5
        progressbar.after(100, progress(currentValue))
        progressbar.update() # Force an update of the GUI
    
    label = Label(win, font=('Helvetica', 20, 'bold'), text="Summary is ready. Enjoy!", background = '#FFFFFF')
    label.grid(column = 1, columnspan=8, row = 6, ipadx = 35, pady = 0, sticky=N+S+E+W)

    
    b = Button(win, text="Close", command=win.destroy, bd=5, relief=RAISED,  font=('Helvetica', 12, 'bold'), bg="white" , fg = '#1da1f2' )
    b.grid(row=8, column=1, columnspan=9, ipadx=5, ipady=5, padx=20, pady=10)
    
    return summary





class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        
# Initialize basic configuration 
        
        self.title("twitterBot_Group12_Studio")
        self.minsize(640, 560)
        self.configure(background = '#FFFFFF')
        self.appHighlightFont = font.Font(family='Helvetica', size=20, weight='bold')
        self.appHighlightFont2 = font.Font(family='Helvetica', size=13, weight='bold')
        self.appHighlightFont3 = font.Font(family='Helvetica', size=16)


# Variable to be changed during 
        
        self.urlString = 'a' # adress to article
        self.filename = '                      ' # article from disk

        self.article = '' # whole article
        self.sumary = '' # best summary
        self.tweet = '' # add # and @ to tweeter
                
#        s = ttk.Style()
#        s.configure('TLabel', foreground='maroon', background= '#1da1f2', font=('Helvetica', 18, 'bold'))
        
        self.url = Entry(self, font=('Helvetica', 12, 'bold'), bd = 4)
        self.url.grid(column=2, columnspan=6, row=3, ipadx=75, ipady=5, padx=5, pady=5)
        self.url.delete(0, END)
        self.url.insert(0, '     Pass article URL here ...')
        
        
#
# Hedder     
        self.hedder = Label(self, background= '#1da1f2', font=('Helvetica', 30, 'bold'), text="  twitterBot Summarizer", foreground='#FFFFFF')
        self.hedder.grid(column = 1, columnspan=8, row = 0, ipadx = 35, pady = 0, sticky=N+S+E+W)
    
        photo = PhotoImage(file=r'C:\Users\Konrad\Desktop\twitter3.png')
        self.hedder2 = Label(self, background= '#1da1f2', font=('Helvetica', 24, 'bold'), image=photo)                     
        self.hedder2.photo = photo
        self.hedder2.grid(column = 9, columnspan=2, row = 0, ipadx = 10, pady = 0, sticky=N+S+E)

#        self.top.title(" Article is being summarized ... ")
#        self.button7 = Button(self.top, text="Dismiss", command=top.destroy)
#        self.button7.grid(column=2, columnspan=6, row=10, ipadx=20, ipady=10, padx=20, pady=10)  
        
#        self.summary = ''           
#        self.labelFrame2 = Label(self, text=self.summary, wraplength = 540, bg = '#e6e6e6', fg='#404040', bd=8,  font=self.appHighlightFont2)
#        self.labelFrame2.grid(column = 1, columnspan=6, row = 7, padx = 10, pady = 4, sticky=N+S+E+W)
        
# Add buttons to APP
        self.button()
        

#        self.radButton = Radiobutton(self, text="Summarize WEB article", value=1, indicatoron=0, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
#        self.radButton.grid(column = 2, columnspan=5, row = 1, ipadx = 10, pady = 0, sticky=N+S+E)
#        self.radButton2 = Radiobutton(self, text="Summarize your own article ", value=2, indicatoron=0, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
#        self.radButton2.grid(column = 7, columnspan=5, row = 1, ipadx = 10, pady = 0, sticky=N+S+E)
#        
        
# ADD MEMU TO APP
        self.menu = Menu(self)
        self.config(menu=self.menu)
        
        self.subMenu = Menu(self.menu)
        self.menu.add_cascade(label="Options", menu=self.subMenu)
        self.subMenu.add_separator()
        self.subMenu.add_command(label="Log In", command=popup_bonus)
        self.subMenu.add_separator()
        self.subMenu.add_command(label="Log Out", command=self.logOut)
        self.subMenu.add_separator()
        self.subMenu.add_command(label="Exit", command=self.destroy)
        
        self.subMenu2 = Menu(self.menu)
        self.menu.add_cascade(label="Summary", menu=self.subMenu2)
        self.subMenu2.add_separator()
        self.subMenu2.add_command(label="Article form disk", command=self.readFile)
        self.subMenu2.add_separator()
        self.subMenu2.add_command(label="Save Summary", command=self.saveSummary)
        self.subMenu2.add_separator()        
        self.subMenu2.add_command(label="Save Article", command=self.saveArticle)
        self.subMenu2.add_separator()



            
# Button definition     
    def button(self):
#        self.buttonReadFile = Button(self, text="Browse a file", command = self.readFile, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
#        self.buttonReadFile.grid(column=1, columnspan=4, row=4, ipadx=20, ipady=10, padx=20, pady=10)  
        
        self.buttonReadURL = Button(self, text="Read URL", command = self.readFromURL, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
        self.buttonReadURL.grid(column=8, columnspan=3, row=3, ipadx=5, ipady=5, padx=5, pady=5)  
        
        self.buttonDisplayArticle = Button(self, text="    Display Article    ", command = self.displayWholeArticle, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
        self.buttonDisplayArticle.grid(column=4, columnspan=4, row=4, ipadx=20, ipady=10, padx=20, pady=10)  
        
        self.buttonAddHastag = Button(self, text="Add Hashtags", command = self.addHashtags, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
        self.buttonAddHastag.grid(column=4, columnspan=4, row=7, ipadx=40, ipady=10, padx=20, pady=10)  
        
        self.buttonPostTweet = Button(self, text="  Post Tweet  ", command = self.postTweet, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="#1da1f2" , fg = 'white' )
        self.buttonPostTweet.grid(column=4, columnspan=4, row=9, ipadx=40, ipady=10, padx=20, pady=10)  
        
        self.buttonSummarize = Button(self, text="  Summarize  ", command = self.summarize_art, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
        self.buttonSummarize.grid(column=4, columnspan=4, row=5, ipadx=40, ipady=10, padx=20, pady=10)  

        self.buttonDisplaySummary = Button(self, text=" Display Summary ", command = self.displayWholeSummary, bd=5, relief=RAISED,  font=self.appHighlightFont2, bg="white" , fg = '#1da1f2' )
        self.buttonDisplaySummary.grid(column=4, columnspan=4, row=6, ipadx=20, ipady=10, padx=20, pady=10)  

# Functions 
    
 
        

    
    def iAmLoggedIn(self):
        pass
    
    def logOut(self):
        showinfo("Info Massage", "You have been successfully logged out!")

    def readFromURL(self):
        self.urlString = self.url.get()
        self.article = get_article(self.urlString)
    
    def readFile(self):
        self.filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetype = (("txt", "*.txt"), ("All Files", "*.*")))
        with open(self.filename, "r") as a:
           self.article = a.read()

        
    def displayWholeArticle(self):
        displayArticle(self.article)    
        
    def displayWholeSummary(self):
        displaySummary(self.summary)

    def summarize_art(self):
        self.summary = summarize_article(self.urlString)
    
    
    
    def addHashtags(self): # check how to it separatly not within summarizer
        self.tweet = twitterize(self.summary).lstrip()
        displaySummary(self.summary)
    
    def postTweet(self):
        make_tweet(self.tweet, self.urlString)
        showinfo("Info Massage", "Your Tweet has been posted!")

    def fileDialog(self):
        pass
     
    def saveSummary(self):
        with open('tweet_summary.txt', 'w') as f:
                f.write("%s," % self.summary)
                
    def saveArticle(self):
        with open('tweet_article.txt', 'w') as f:
                f.write("%s," % self.article)
    
    
# MAIN LOOP

if __name__ == '__main__':
    root = Root()
    root.mainloop()