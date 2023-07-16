from flask import Flask, request, render_template
import tkinter as tk
from tkinter import messagebox
from pyrfc import *
import mysql.connector

@app.route('/')
def my_form():
    return render_template('SapUserPasswordReset.html')

@app.route('/', methods=['POST'])
def my_form_post():
    variable = request.form['user_reset']
    return variable


#If you have a large number of widgets, like it looks like you will for your
#game you can specify the attributes for all widgets simply like this.



#You can set the geometry attribute to change the root windows size


#Changed variables so you don't have these set to None from .pack()


