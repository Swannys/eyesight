# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:51:38 2024

@author: Brendan
"""


import wx
import os
from scratch import EyesightManager
from data import Prediction_FXN, Label_FXN,  plot_cm, ROC
from joblib import load

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, roc_curve, roc_auc_score
from itertools import product
# from prep import get_split_lists

import tensorflow as tf 
from tensorflow import keras
from IPython.display import Image, display


class EyesightGUI ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Eyesight GUI - VGG16 Based Convolutional Neural Network from Scratch", pos = wx.DefaultPosition, size = wx.Size( 1400,600 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        bSizer1 = wx.BoxSizer( wx.VERTICAL )

        gSizer3 = wx.GridSizer( 0, 6, 5, 5 )

        bSizer8 = wx.BoxSizer( wx.VERTICAL )

        gSizer7 = wx.GridSizer( 2, 0, 0, 0 )
        

        bSizer16 = wx.BoxSizer( wx.VERTICAL )
        

        #sbSizer9 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Data Selection" ), wx.VERTICAL )

        #self.m_staticText361 = wx.StaticText( sbSizer9.GetStaticBox(), wx.ID_ANY, u"Creator", wx.DefaultPosition, wx.DefaultSize, 0 )
        #self.m_staticText361.Wrap( -1 )

        #sbSizer9.Add( self.m_staticText361, 0, wx.ALL, 5 )

        #self.m_dataset_enter = wx.TextCtrl( sbSizer9.GetStaticBox(), wx.ID_ANY, 'Testing3_Dataset_packed', wx.DefaultPosition, wx.DefaultSize, 0 )
        #sbSizer9.Add( self.m_dataset_enter, 0, wx.ALL, 5 )


        #bSizer16.Add( sbSizer9, 1, wx.EXPAND, 5 )


        #gSizer7.Add( bSizer16, 1, wx.EXPAND, 5 )

        bSizer17 = wx.BoxSizer( wx.VERTICAL )

        sbSizer10 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Optimizer and Loss Function" ), wx.VERTICAL )

        gSizer8 = wx.GridSizer( 2, 0, 0, 0 )

        sbSizer12 = wx.StaticBoxSizer( wx.StaticBox( sbSizer10.GetStaticBox(), wx.ID_ANY, u"Optimizer" ), wx.VERTICAL )

        Optimizer_ChoiceChoices = [ u"Nadam", u"Adam", u"SGD" ]
        self.Optimizer_Choice = wx.Choice( sbSizer12.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, Optimizer_ChoiceChoices, 0 )
        self.Optimizer_Choice.SetSelection( 0 )
        sbSizer12.Add( self.Optimizer_Choice, 0, wx.ALL, 5 )

        self.m_staticText42 = wx.StaticText( sbSizer12.GetStaticBox(), wx.ID_ANY, u"Learning Rate:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText42.Wrap( -1 )

        sbSizer12.Add( self.m_staticText42, 0, wx.ALL, 5 )

        self.m_lr = wx.TextCtrl( sbSizer12.GetStaticBox(), wx.ID_ANY, u"0.001", wx.DefaultPosition, wx.DefaultSize, 0 )
        sbSizer12.Add( self.m_lr, 0, wx.ALL, 5 )


        gSizer8.Add( sbSizer12, 1, wx.EXPAND, 5 )

        sbSizer13 = wx.StaticBoxSizer( wx.StaticBox( sbSizer10.GetStaticBox(), wx.ID_ANY, u"Loss Function" ), wx.VERTICAL )

        Loss_ChoiceChoices = [ u"Categorical Cross Entropy" ]
        self.Loss_Choice = wx.Choice( sbSizer13.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, Loss_ChoiceChoices, 0 )
        self.Loss_Choice.SetSelection( 0 )
        sbSizer13.Add( self.Loss_Choice, 0, wx.ALL, 5 )


        gSizer8.Add( sbSizer13, 1, wx.EXPAND, 5 )


        sbSizer10.Add( gSizer8, 1, wx.EXPAND, 5 )


        bSizer17.Add( sbSizer10, 1, wx.EXPAND, 5 )


        gSizer7.Add( bSizer17, 1, wx.EXPAND, 5 )


        bSizer8.Add( gSizer7, 1, wx.EXPAND, 5 )


        gSizer3.Add( bSizer8, 1, wx.EXPAND, 5 )

        bSizer9 = wx.BoxSizer( wx.VERTICAL )

        gSizer71 = wx.GridSizer( 3, 0, 0, 0 )


        sbSizer131 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Model Selection" ), wx.VERTICAL )

        fgSizer1 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer1.SetFlexibleDirection( wx.BOTH )
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )


        fgSizer1.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.m_staticText7 = wx.StaticText( sbSizer131.GetStaticBox(), wx.ID_ANY, u"Name:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText7.Wrap( -1 )

        fgSizer1.Add( self.m_staticText7, 0, wx.ALL, 5 )

        self.new_mod = wx.Button( sbSizer131.GetStaticBox(), wx.ID_ANY, u"New Model", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer1.Add( self.new_mod, 0, wx.ALL, 5 )

        model_optionsChoices = [ u"Eyesight" ]
        self.model_options = wx.Choice( sbSizer131.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, model_optionsChoices, 0 )
        self.model_options.SetSelection( 0 )
        fgSizer1.Add( self.model_options, 0, wx.ALL, 5 )

        self.load_mod = wx.Button( sbSizer131.GetStaticBox(), wx.ID_ANY, u"Load Model", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer1.Add( self.load_mod, 0, wx.ALL, 5 )

        self.load_name = wx.TextCtrl( sbSizer131.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer1.Add( self.load_name, 0, wx.ALL, 5 )

        self.save_mod = wx.Button( sbSizer131.GetStaticBox(), wx.ID_ANY, u"Save Model", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer1.Add( self.save_mod, 0, wx.ALL, 5 )

        self.save_name = wx.TextCtrl( sbSizer131.GetStaticBox(), wx.ID_ANY, "Unnamed", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer1.Add( self.save_name, 0, wx.ALL, 5 )


        sbSizer131.Add( fgSizer1, 1, wx.EXPAND, 5 )


        gSizer71.Add( sbSizer131, 1, wx.EXPAND, 5 )

        sbSizer121 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Model Details" ), wx.VERTICAL )

        fgSizer3 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer3.SetFlexibleDirection( wx.BOTH )
        fgSizer3.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

        self.m_staticText4 = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"Name", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4.Wrap( -1 )

        fgSizer3.Add( self.m_staticText4, 0, wx.ALL, 5 )

        self.mod_name = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"Eyesight", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.mod_name.Wrap( -1 )

        fgSizer3.Add( self.mod_name, 0, wx.ALL, 5 )

        self.m_staticText5 = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"Conv Layers", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText5.Wrap( -1 )

        fgSizer3.Add( self.m_staticText5, 0, wx.ALL, 5 )

        self.conv_layers = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"13", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.conv_layers.Wrap( -1 )

        fgSizer3.Add( self.conv_layers, 0, wx.ALL, 5 )

        self.m_staticText6 = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"FC Layers", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText6.Wrap( -1 )

        fgSizer3.Add( self.m_staticText6, 0, wx.ALL, 5 )

        self.fc_layers = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"3", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.fc_layers.Wrap( -1 )

        fgSizer3.Add( self.fc_layers, 0, wx.ALL, 5 )

        self.Epochs_num = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"Epochs:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Epochs_num.Wrap( -1 )

        fgSizer3.Add( self.Epochs_num, 0, wx.ALL, 5 )

        self.mod_epochs = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"Unknown", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.mod_epochs.Wrap( -1 )

        fgSizer3.Add( self.mod_epochs, 0, wx.ALL, 5 )

        self.Busy_Flag = wx.StaticText( sbSizer121.GetStaticBox(), wx.ID_ANY, u"Model Busy", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Busy_Flag.Wrap( -1 )

        self.Busy_Flag.SetForegroundColour( wx.Colour( 255, 0, 0 ) )
        self.Busy_Flag.Hide()

        fgSizer3.Add( self.Busy_Flag, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 5 )


        sbSizer121.Add( fgSizer3, 1, wx.EXPAND, 5 )


        gSizer71.Add( sbSizer121, 1, wx.EXPAND, 5 )

        sbSizer14 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Using Model" ), wx.VERTICAL )

        fgSizer2 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer2.SetFlexibleDirection( wx.BOTH )
        fgSizer2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

        self.train_mod = wx.Button( sbSizer14.GetStaticBox(), wx.ID_ANY, u"Train", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer2.Add( self.train_mod, 0, wx.ALL, 5 )

        self.test_mod = wx.Button( sbSizer14.GetStaticBox(), wx.ID_ANY, u"Test", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer2.Add( self.test_mod, 0, wx.ALL, 5 )

        self.m_staticText1 = wx.StaticText( sbSizer14.GetStaticBox(), wx.ID_ANY, u"Epochs:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText1.Wrap( -1 )

        fgSizer2.Add( self.m_staticText1, 0, wx.ALL, 5 )

        self.m_epoch_set = wx.TextCtrl( sbSizer14.GetStaticBox(), wx.ID_ANY, u"10", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer2.Add( self.m_epoch_set, 0, wx.ALL, 5 )

        self.m_staticText2 = wx.StaticText( sbSizer14.GetStaticBox(), wx.ID_ANY, u"Batch:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText2.Wrap( -1 )

        fgSizer2.Add( self.m_staticText2, 0, wx.ALL, 5 )

        self.m_batch_set = wx.TextCtrl( sbSizer14.GetStaticBox(), wx.ID_ANY, u"10", wx.DefaultPosition, wx.DefaultSize, 0 )
        fgSizer2.Add( self.m_batch_set, 0, wx.ALL, 5 )


        sbSizer14.Add( fgSizer2, 1, wx.EXPAND, 5 )


        gSizer71.Add( sbSizer14, 1, wx.EXPAND, 5 )


        bSizer9.Add( gSizer71, 1, wx.EXPAND, 5 )


        gSizer3.Add( bSizer9, 0, wx.EXPAND, 5 )

        bSizer7 = wx.BoxSizer( wx.VERTICAL )

        gSizer81 = wx.GridSizer( 2, 0, 0, 0 )

        sbSizer15 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Bus" ), wx.VERTICAL )

        bSizer112 = wx.BoxSizer( wx.VERTICAL )

        self.heatmap_nc = wx.StaticBitmap( sbSizer15.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer112.Add( self.heatmap_nc, 0, wx.ALL, 5 )


        sbSizer15.Add( bSizer112, 1, wx.EXPAND, 5 )

        bSizer12 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText37 = wx.StaticText( sbSizer15.GetStaticBox(), wx.ID_ANY, u"Classified as:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText37.Wrap( -1 )

        bSizer12.Add( self.m_staticText37, 0, wx.ALL, 5 )

        self.Bus_guess = wx.StaticText( sbSizer15.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Bus_guess.Wrap( -1 )

        bSizer12.Add( self.Bus_guess, 0, wx.ALL, 5 )


        sbSizer15.Add( bSizer12, 0, wx.EXPAND, 5 )


        gSizer81.Add( sbSizer15, 1, wx.EXPAND, 5 )

        sbSizer16 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Training" ), wx.VERTICAL )

        bSizer111 = wx.BoxSizer( wx.VERTICAL )

        self.m_Training_MTX_ROC = wx.StaticBitmap( sbSizer16.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer111.Add( self.m_Training_MTX_ROC, 0, wx.ALL, 5 )


        sbSizer16.Add( bSizer111, 1, wx.EXPAND, 5 )

        self.B_Train_perf = wx.Button( sbSizer16.GetStaticBox(), wx.ID_ANY, u"View ROC", wx.DefaultPosition, wx.DefaultSize, 0 )
        sbSizer16.Add( self.B_Train_perf, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        gSizer81.Add( sbSizer16, 1, wx.EXPAND, 5 )


        bSizer7.Add( gSizer81, 1, wx.EXPAND, 5 )


        gSizer3.Add( bSizer7, 1, wx.EXPAND, 5 )

        bSizer10 = wx.BoxSizer( wx.VERTICAL )

        gSizer9 = wx.GridSizer( 2, 0, 0, 0 )

        sbSizer17 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Car" ), wx.VERTICAL )

        bSizer14 = wx.BoxSizer( wx.VERTICAL )

        self.heatmap_nccl = wx.StaticBitmap( sbSizer17.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer14.Add( self.heatmap_nccl, 0, wx.ALL, 5 )


        sbSizer17.Add( bSizer14, 1, wx.EXPAND, 5 )

        bSizer121 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText371 = wx.StaticText( sbSizer17.GetStaticBox(), wx.ID_ANY, u"Classified as:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText371.Wrap( -1 )

        bSizer121.Add( self.m_staticText371, 0, wx.ALL, 5 )

        self.Car_guess = wx.StaticText( sbSizer17.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Car_guess.Wrap( -1 )

        bSizer121.Add( self.Car_guess, 0, wx.ALL, 5 )


        sbSizer17.Add( bSizer121, 0, wx.EXPAND, 5 )


        gSizer9.Add( sbSizer17, 1, wx.EXPAND, 5 )

        sbSizer18 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Testing" ), wx.VERTICAL )

        bSizer91 = wx.BoxSizer( wx.VERTICAL )

        self.m_Testing_MTX_ROC = wx.StaticBitmap( sbSizer18.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer91.Add( self.m_Testing_MTX_ROC, 0, wx.ALL, 5 )


        sbSizer18.Add( bSizer91, 1, wx.EXPAND, 5 )

        self.b_Test_perf = wx.Button( sbSizer18.GetStaticBox(), wx.ID_ANY, u"View ROC", wx.DefaultPosition, wx.DefaultSize, 0 )
        sbSizer18.Add( self.b_Test_perf, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


        gSizer9.Add( sbSizer18, 1, wx.EXPAND, 5 )


        bSizer10.Add( gSizer9, 1, wx.EXPAND, 5 )


        gSizer3.Add( bSizer10, 1, wx.EXPAND, 5 )

        bSizer11 = wx.BoxSizer( wx.VERTICAL )

        gSizer10 = wx.GridSizer( 2, 0, 0, 0 )

        sbSizer19 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Person" ), wx.VERTICAL )

        bSizer18 = wx.BoxSizer( wx.VERTICAL )

        self.heatmap_rc = wx.StaticBitmap( sbSizer19.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer18.Add( self.heatmap_rc, 0, wx.ALL, 5 )


        sbSizer19.Add( bSizer18, 1, wx.EXPAND, 5 )

        bSizer1211 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText3711 = wx.StaticText( sbSizer19.GetStaticBox(), wx.ID_ANY, u"Classified as:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText3711.Wrap( -1 )

        bSizer1211.Add( self.m_staticText3711, 0, wx.ALL, 5 )

        self.Person_guess = wx.StaticText( sbSizer19.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Person_guess.Wrap( -1 )

        bSizer1211.Add( self.Person_guess, 0, wx.ALL, 5 )

        
        sbSizer19.Add( bSizer1211, 0, wx.EXPAND, 5 )


        gSizer10.Add( sbSizer19, 1, wx.EXPAND, 5 )
        
        
        
        bSizer10000 = wx.BoxSizer( wx.VERTICAL )

        gSizer9000 = wx.GridSizer( 2, 0, 0, 0 )

        sbSizer17000 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Motorcycle" ), wx.VERTICAL )

        bSizer14000 = wx.BoxSizer( wx.VERTICAL )

        self.heatmap_nccl = wx.StaticBitmap( sbSizer17000.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
        bSizer14000.Add( self.heatmap_nccl, 0, wx.ALL|wx.EXPAND, 5 )


        sbSizer17000.Add( bSizer14, 1, wx.EXPAND, 5 )

        bSizer121000 = wx.BoxSizer( wx.HORIZONTAL )

        self.m_staticText371 = wx.StaticText( sbSizer17000.GetStaticBox(), wx.ID_ANY, u"Classified as:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText371.Wrap( -1 )

        bSizer121000.Add( self.m_staticText371, 0, wx.ALL, 5 )

        self.Motorcycle_guess = wx.StaticText( sbSizer17000.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Motorcycle_guess.Wrap( -1 )

        bSizer121000.Add( self.Motorcycle_guess, 0, wx.ALL, 5 )


        sbSizer17000.Add( bSizer121000, 0, wx.EXPAND, 5 )


        gSizer9000.Add( sbSizer17000, 1, wx.EXPAND, 5 )


        bSizer10000.Add( gSizer9000, 1, wx.EXPAND, 5 )


        gSizer3.Add( bSizer10000, 1, wx.EXPAND, 5 )
           
        
        
        
        
        
        

        sbSizer20 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Performance Parameters" ), wx.VERTICAL )

        gSizer82 = wx.GridSizer( 10, 3, 0, 0 )

        self.m_staticText25 = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Train", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText25.Wrap( -1 )

        gSizer82.Add( self.m_staticText25, 0, wx.ALL, 5 )

        self.train_acc_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Accuracy:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_acc_label.Wrap( -1 )

        gSizer82.Add( self.train_acc_label, 0, wx.ALL, 5 )

        self.train_acc = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_acc.Wrap( -1 )

        gSizer82.Add( self.train_acc, 0, wx.ALL, 5 )

        bSizer171 = wx.BoxSizer( wx.VERTICAL )

        Train_param_classChoices = [ u"Bus", u"Car", u"Motorcycle", u"Person" ]
        self.Train_param_class = wx.Choice( sbSizer20.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, Train_param_classChoices, 0 )
        self.Train_param_class.SetSelection( 0 )
        bSizer171.Add( self.Train_param_class, 1, wx.ALL, 2 )


        gSizer82.Add( bSizer171, 1, wx.EXPAND, 5 )

        self.train_sens_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Sensitivity:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_sens_label.Wrap( -1 )

        gSizer82.Add( self.train_sens_label, 0, wx.ALL, 5 )

        self.train_sens = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_sens.Wrap( -1 )

        gSizer82.Add( self.train_sens, 0, wx.ALL, 5 )


        gSizer82.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.train_spef_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Specificity:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_spef_label.Wrap( -1 )

        gSizer82.Add( self.train_spef_label, 0, wx.ALL, 5 )

        self.train_spef = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_spef.Wrap( -1 )

        gSizer82.Add( self.train_spef, 0, wx.ALL, 5 )


        gSizer82.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.train_ppv_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"PPV:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_ppv_label.Wrap( -1 )

        gSizer82.Add( self.train_ppv_label, 0, wx.ALL, 5 )

        self.train_ppv = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_ppv.Wrap( -1 )

        gSizer82.Add( self.train_ppv, 0, wx.ALL, 5 )


        gSizer82.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.train_npv_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"NPV:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_npv_label.Wrap( -1 )

        gSizer82.Add( self.train_npv_label, 0, wx.ALL, 5 )

        self.train_npv = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.train_npv.Wrap( -1 )

        self.train_npv.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ) )

        gSizer82.Add( self.train_npv, 0, wx.ALL, 5 )

        self.m_staticText36 = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Test", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText36.Wrap( -1 )

        gSizer82.Add( self.m_staticText36, 0, wx.ALL, 5 )

        self.test_acc_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Accuracy:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_acc_label.Wrap( -1 )

        gSizer82.Add( self.test_acc_label, 0, wx.ALL, 5 )

        self.test_acc = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_acc.Wrap( -1 )

        self.test_acc.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ) )

        gSizer82.Add( self.test_acc, 0, wx.ALL, 5 )

        bSizer1711 = wx.BoxSizer( wx.VERTICAL )

        Test_param_classChoices = [ u"Bus", u"Car", u"Motorcycle", u"Person" ]
        self.Test_param_class = wx.Choice( sbSizer20.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, Test_param_classChoices, 0 )
        self.Test_param_class.SetSelection( 0 )
        bSizer1711.Add( self.Test_param_class, 0, wx.ALL, 2 )


        gSizer82.Add( bSizer1711, 1, wx.EXPAND, 5 )

        self.test_sens_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Sensitivity:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_sens_label.Wrap( -1 )

        gSizer82.Add( self.test_sens_label, 0, wx.ALL, 5 )

        self.test_sens = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_sens.Wrap( -1 )

        self.test_sens.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ) )

        gSizer82.Add( self.test_sens, 0, wx.ALL, 5 )


        gSizer82.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.test_spef_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"Specificity:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_spef_label.Wrap( -1 )

        gSizer82.Add( self.test_spef_label, 0, wx.ALL, 5 )

        self.test_spef = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_spef.Wrap( -1 )

        self.test_spef.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ) )

        gSizer82.Add( self.test_spef, 0, wx.ALL, 5 )


        gSizer82.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.test_ppv_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"PPV:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_ppv_label.Wrap( -1 )

        gSizer82.Add( self.test_ppv_label, 0, wx.ALL, 5 )

        self.test_ppv = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_ppv.Wrap( -1 )

        gSizer82.Add( self.test_ppv, 0, wx.ALL, 5 )


        gSizer82.Add( ( 0, 0), 1, wx.EXPAND, 5 )

        self.test_npv_label = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"NPV:", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_npv_label.Wrap( -1 )

        gSizer82.Add( self.test_npv_label, 0, wx.ALL, 5 )

        self.test_npv = wx.StaticText( sbSizer20.GetStaticBox(), wx.ID_ANY, u"0%", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.test_npv.Wrap( -1 )

        gSizer82.Add( self.test_npv, 0, wx.ALL, 5 )


        sbSizer20.Add( gSizer82, 1, wx.EXPAND, 5 )


        gSizer10.Add( sbSizer20, 1, wx.EXPAND, 5 )


        bSizer11.Add( gSizer10, 1, wx.EXPAND, 5 )


        gSizer3.Add( bSizer11, 1, wx.EXPAND, 5 )


        bSizer1.Add( gSizer3, 1, wx.EXPAND, 5 )


        self.SetSizer( bSizer1 )
        self.Layout()

        self.Centre( wx.BOTH )

        self.Additional_initialization()

    def __del__( self ):
        self.NM.reset_model()
        self.NM = None

    def Additional_initialization(self):
        #For Global Flags, vars and event binds. Add to __init__ after transfer.
        
        self.DataFull = False
        self.ModelBusy = False
        self.plot_complete = False
        self.NM = EyesightManager()
        
        self.train_on_roc = True
        self.test_on_roc = True
        
        self.new_mod.Bind( wx.EVT_BUTTON, self.new_model_selected)
        self.load_mod.Bind( wx.EVT_BUTTON, self.load_model_selected)

        self.save_mod.Bind( wx.EVT_BUTTON, self.save_model_selected)

        self.train_mod.Bind( wx.EVT_BUTTON, self.train_button)
        self.test_mod.Bind( wx.EVT_BUTTON, self.test_button)

        self.B_Train_perf.Bind( wx.EVT_BUTTON, self.train_param_view )
        self.b_Test_perf.Bind( wx.EVT_BUTTON, self.test_param_view )
        
        self.Train_param_class.Bind( wx.EVT_CHOICE, self.update_params )
        self.Test_param_class.Bind( wx.EVT_CHOICE, self.update_params )
        
        #File Management Code:
        self.tmp_dir =  'E:\Brendan_2023_24\dump_tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)


    def mod_set_busy(self):
        self.ModelBusy = True
        self.Busy_Flag.Show()

    def mod_set_clear(self):
        self.ModelBusy = False
        self.Busy_Flag.Hide()

    def train_button(self,event):
        self.NM.rename_model(self.save_name.GetLineText(0))
        self.mod_set_busy()
        self.NM.re_compile_model(opt=int(self.Optimizer_Choice.GetSelection()), lr = float(self.m_lr.GetLineText(0)))
        Epochs = int(self.m_epoch_set.GetLineText(0))
        Batch = int(self.m_batch_set.GetLineText(0))
        Study = 'Dataset/Packed'
        self.mod_epochs.SetLabel(str(Epochs))

        self.NM.train_model(epochs = Epochs, batch_size = Batch, dataset = Study)
            
        self.update_params(event)
        self.mod_set_clear()

    def test_button(self,event):
        self.NM.rename_model(self.save_name.GetLineText(0))
        self.mod_set_clear()
        self.update_params(event)
        self.NM.display()

    def new_model_selected(self,event):
        self.plot_complete = False
        model_list = ['Eyesight']
        selection = self.model_options.GetSelection()
        
        self.NM.rename_model(self.save_name.GetLineText(0))
        self.clear_perfs()

        self.NM.reset_model()
        if selection == 0:
            print('Building Eyesight')
            self.NM.build_eyesight()
            print('Build complete')
            self.update_model_details('Eyesight','13','3','0')
            self.save_name.SetValue("Eyesight_unnamed")
        else:
            self.NM.reset_model()
            self.update_model_details('Eyesight', '13', '3', '0')
        self.NM.re_compile_model(opt=int(self.Optimizer_Choice.GetSelection()), lr = float(self.m_lr.GetLineText(0)))

    def load_model_selected(self,event):
        self.plot_complete = False
        load_name = self.load_name.GetLineText(0)
        self.save_name.SetValue(load_name)
        self.NM.rename_model(load_name)
        self.update_model_details(load_name,'13','3','50')
        try:
            self.NM.reset_model()
            self.NM.load_model(f"E:\Brendan_2023_24\CNN\CNN_Saved_{load_name}.h5")
            self.clear_perfs()
        except Exception as e:
            print(f'Model not found: {e}')

    def save_model_selected(self,event):
        '''
        If an unused name is entered, the .h5 version of the currently loaded model
        will be saved
        '''
        if self.NM.model:
            new_name = self.save_name.GetLineText(0)
            self.NM.model.save(f"E:\Brendan_2023_24\CNN\CNN_Saved_{new_name}.h5")
        else:
            print('No Model to save')

    def train_param_view(self,event):
        '''switch between ROC and CFN MTX'''
        if self.NM.model:
            if self.train_on_roc:
                self.m_Training_MTX_ROC.SetBitmap(self.im_format_png('E:\Brendan_2023_24\dump_tmp/Training_roc.png'))
                self.train_on_roc = False
                self.B_Train_perf.SetLabel('View CM')
                self.Refresh()
            else:
                self.m_Training_MTX_ROC.SetBitmap(self.im_format_png('E:\Brendan_2023_24\dump_tmp/Training_cm.png'))
                self.train_on_roc = True
                self.B_Train_perf.SetLabel('View ROC')
                self.Refresh()

    def test_param_view(self,event):
        '''switch between ROC and CFN MTX'''
        if self.NM.model:
            if self.test_on_roc:
                self.m_Testing_MTX_ROC.SetBitmap(self.im_format_png('E:\Brendan_2023_24\dump_tmp/Testing_roc.png'))
                self.test_on_roc = False
                self.b_Test_perf.SetLabel('View CM')
                self.Refresh()
            else:
                self.m_Testing_MTX_ROC.SetBitmap(self.im_format_png('E:\Brendan_2023_24\dump_tmp/Testing_cm.png'))
                self.test_on_roc = True
                self.b_Test_perf.SetLabel('View ROC')
                self.Refresh()
    def update_model_details(self, name = 'Eyesight', conv = '13', fc = '3', epochs = 'No Model'):
        '''
            Show name, conv layers, fc layers, and epochs of a model
        '''
        self.mod_name.SetLabel(name)
        self.conv_layers.SetLabel(conv)
        self.fc_layers.SetLabel(fc)
        self.mod_epochs.SetLabel(epochs)



    def update_params(self,event):
        '''
            If there are trained models available and one is selected,
            This function will display the performance parameters of the models
        '''
        if self.NM.model:
            model = self.NM.name
            user = 'Brendan'
            classes = ["Bus", "Car", "Motorcycles", "Person"]
            
            dump_dir = 'E:\Brendan_2023_24\dump_tmp'
        
            Training_Label_Predict = load(f"E:\{user}_2023_24\CNN\{model}\Training_Label_Predict.sav")
            Label_train = load(f"E:\{user}_2023_24\Dataset\Packed\Label_train.sav")
            
            
            Training_Label_Predict_FXN = Prediction_FXN(Training_Label_Predict)
            Label_train_FNX = Label_FXN(Label_train)
            tr_cm = confusion_matrix(Label_train_FNX, Training_Label_Predict_FXN, labels = classes)
            tr_multi_cm = multilabel_confusion_matrix(Label_train_FNX, Training_Label_Predict_FXN, labels = classes)
            
            if not self.plot_complete:
                plot_cm(tr_cm, data = "Training", save_dir = dump_dir)
                ROC(Label_train, Training_Label_Predict, "Training", classes, tr_multi_cm, save_dir = dump_dir)
                self.m_Training_MTX_ROC.SetBitmap(self.im_format_png(f'{dump_dir}/Training_cm.png'))
                
            self.result_update(tr_multi_cm[self.Train_param_class.GetSelection()], "Training")
    
            
            Testing_Label_Predict = load(f"E:\{user}_2023_24\CNN\{model}\Testing_Label_Predict.sav")
            Label_test = load(f"E:\{user}_2023_24\Dataset\Packed\Label_test.sav")
            
            
            Testing_Label_Predict_FXN = Prediction_FXN(Testing_Label_Predict)
            Label_test_FXN = Label_FXN(Label_test)
            te_cm = confusion_matrix(Label_test_FXN, Testing_Label_Predict_FXN, labels = classes)
            te_multi_cm = multilabel_confusion_matrix(Label_test_FXN, Testing_Label_Predict_FXN, labels = classes)
            if not self.plot_complete:
                plot_cm(te_cm, data = "Testing", save_dir = dump_dir)
                ROC(Label_test, Testing_Label_Predict, "Testing", classes, te_multi_cm, save_dir = dump_dir)
                #self.heatmap_gen()
                #self.heatmap_bus.SetBitmap(self.im_format_png(f'{dump_dir}/bus_hm.png'))
                #self.heatmap_car.SetBitmap(self.im_format_png(f'{dump_dir}/car_hm.png'))
                #self.heatmap_motorcycle.SetBitmap(self.im_format_png(f'{dump_dir}/motorcycle_hm.png'))
                #self.heatmap_person.SetBitmap(self.im_format_png(f'{dump_dir}/person_hm.png'))
                self.m_Testing_MTX_ROC.SetBitmap(self.im_format_png(f'{dump_dir}/Testing_cm.png'))
                self.plot_complete = True
                
            self.result_update(te_multi_cm[self.Test_param_class.GetSelection()], "Testing")
            self.Refresh()
            
            
            
        else:
            print('No Model Loaded')
            self.clear_perfs()
        

        
    def result_update(self, multi_cm, data):
        try:
            TN, FP, FN, TP = multi_cm.ravel()
            AUC_Score = float(TP + TN) / float(TP + TN + FP + FN)
            Sensitivity = float(TP / float(TP + FN))
            Specificity = float(TN / float(TN + FP))
            PPV = float(TP / float(TP + FP))
            NPV = float(TN / float(TN + FN))
            if data =="Training":
                self.train_acc.SetLabel(f'{AUC_Score*100:.2f}%')
                self.train_sens.SetLabel(f'{Sensitivity*100:.2f}%')
                self.train_spef.SetLabel(f'{Specificity*100:.2f}%')
                self.train_ppv.SetLabel(f'{PPV*100:.2f}%')
                self.train_npv.SetLabel(f'{NPV*100:.2f}%')
                self.train_acc.SetBackgroundColour( self.get_colouuuuur(AUC_Score) )
                self.train_sens.SetBackgroundColour( self.get_colouuuuur(Sensitivity) )
                self.train_spef.SetBackgroundColour( self.get_colouuuuur(Specificity) )
                self.train_ppv.SetBackgroundColour( self.get_colouuuuur(PPV) )
                self.train_npv.SetBackgroundColour( self.get_colouuuuur(NPV) )
            if data =="Testing":
                self.test_acc.SetLabel(f'{AUC_Score*100:.2f}%')
                self.test_sens.SetLabel(f'{Sensitivity*100:.2f}%')
                self.test_spef.SetLabel(f'{Specificity*100:.2f}%')
                self.test_ppv.SetLabel(f'{PPV*100:.2f}%')
                self.test_npv.SetLabel(f'{NPV*100:.2f}%')
                self.test_acc.SetBackgroundColour( self.get_colouuuuur(AUC_Score) )
                self.test_sens.SetBackgroundColour( self.get_colouuuuur(Sensitivity) )
                self.test_spef.SetBackgroundColour( self.get_colouuuuur(Specificity) )
                self.test_ppv.SetBackgroundColour( self.get_colouuuuur(PPV) )
                self.test_npv.SetBackgroundColour( self.get_colouuuuur(NPV) )
            
        except Exception as e:
            print(f'Model is not trained or: {e}')
            self.clear_perfs()
    def get_colouuuuur(self, stats):
        if stats > 0.95:
            coloueueur = (0,174,0)
        elif stats > 0.87:
            coloueueur = (128,255,0)
        elif stats > 0.75:
            coloueueur = (255,255,0)
        else:
            coloueueur = (255,0,0)
        return coloueueur
            
    
    def clear_perfs(self):
        self.train_acc.SetLabel(f'NA')
        self.train_sens.SetLabel(f'NA')
        self.train_spef.SetLabel(f'NA')
        self.train_ppv.SetLabel(f'NA')
        self.train_npv.SetLabel(f'NA')
        self.test_acc.SetLabel(f'NA')
        self.test_sens.SetLabel(f'NA')
        self.test_spef.SetLabel(f'NA')
        self.test_ppv.SetLabel(f'NA')
        self.test_npv.SetLabel(f'NA')
        
        self.train_acc.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.train_sens.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.train_spef.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.train_ppv.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.train_npv.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.test_acc.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.test_sens.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.test_spef.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.test_ppv.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.test_npv.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        
        self.m_Training_MTX_ROC.SetBitmap( wx.NullBitmap )
        self.m_Testing_MTX_ROC.SetBitmap( wx.NullBitmap )
        self.heatmap_nc.SetBitmap( wx.NullBitmap )
        self.heatmap_nccl.SetBitmap( wx.NullBitmap )
        self.heatmap_rc.SetBitmap( wx.NullBitmap )
        
        self.Bus_guess.SetLabel(f'NA')
        self.Car_guess.SetLabel(f'NA')
        self.Motorcycle_guess.SetLabel(f'NA')
        self.Person_guess.SetLabel(f'NA')
        
        self.Bus_guess.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.Car_guess.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.Motorcycle_guess.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        self.Person_guess.SetBackgroundColour(wx.SystemSettings.GetColour( wx.SYS_COLOUR_APPWORKSPACE ))
        
        self.B_Train_perf.SetLabel('View ROC')
        self.b_Test_perf.SetLabel('View ROC')
        
        

    def im_format_png(self,dir_to_png):
        '''returns a bitmap image to be displayed in GUI'''
        im = wx.Image(dir_to_png, wx.BITMAP_TYPE_PNG )
        im = im.Scale(225,180,wx.IMAGE_QUALITY_HIGH)
        return wx.Bitmap(im)
        
        
    def heatmap_gen(self, path_to_src=None, hm_class=None):
        img_size = (244, 244)
        last_conv_layer_name = "last_layer"
        
        
        def get_img_array(img_path, size):
            # `img`image of size 244x244
            img = keras.preprocessing.image.load_img(img_path, target_size=size)
            # `array` is a float32 Numpy array of shape (244, 244, 3)
            array = keras.preprocessing.image.img_to_array(img)
            # We add a dimension to transform our array into a "batch"
            # of size (1, 299, 299, 3)
            array = np.expand_dims(array, axis=0)
            return array
        
        def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
            # First, we create a model that maps the input image to the activations
            # of the last conv layer as well as the output predictions
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
        
            # Then, we compute the gradient of the top predicted class for our input image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]
        
            # This is the gradient of the output neuron (top predicted or chosen)
            # with regard to the output feature map of the last conv layer
            grads = tape.gradient(class_channel, last_conv_layer_output)
        
            # This is a vector where each entry is the mean intensity of the gradient
            # over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
            # We multiply each channel in the feature map array
            # by "how important this channel is" with regard to the top predicted class
            # then sum all the channels to obtain the heatmap class activation
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
        
            # For visualization purpose, we will also normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
        
        
        im_bus = 'E:/Brendan_2023_24/Dataset/Processed/Bus/image_0_85_flipped.png'
        im_car = 'E:/Brendan_2023_24/Dataset/Processed/Car/image_1_5_flipped.png'
        im_motorcycle = 'E:/Brendan_2023_24/Dataset/Processed/Motorcycles/image_2_26.png'
        im_person = 'E:/Brendan_2023_24/Dataset/Processed/Person/image_3_65.png'
        
        # Prepare image
        img_array_bus = get_img_array(im_bus, size=img_size)
        img_array_car = get_img_array(im_car, size=img_size)
        img_array_motorcycles = get_img_array(im_motorcycle, size=img_size)
        img_array_person = get_img_array(im_person, size=img_size)
        
        # Make model
        model = self.NM.model

        
        def decode_pred(preds):
            classes = ['Bus', 'Car', 'Motorcycles', 'Person']
            max_idx = None
            max_value = None
            for idx, num in enumerate(preds):
                if (max_value is None or num > max_value):
                    max_idx = idx
                    max_value = num
            return classes[max_idx]
        # Remove last layer's softmax
        model.layers[-1].activation = None
        # Print what the top predicted class is
        preds = model.predict(img_array_bus)
        #print("Predicted:", decode_pred(preds))
        self.Bus_guess.SetLabel(decode_pred(preds))
        if decode_pred(preds) == 'Bus':
            self.Bus_guess.SetBackgroundColour((0,255,0))
        else:
            self.Bus_guess.SetBackgroundColour((255,0,0))
        
        preds = model.predict(img_array_car)
        self.Car_guess.SetLabel(decode_pred(preds))
        if decode_pred(preds) == 'Car':
            self.Car_guess.SetBackgroundColour((0,255,0))
        else:
            self.Car_guess.SetBackgroundColour((255,0,0))
            
        preds = model.predict(img_array_motorcycles)
        self.Motorcycle_guess.SetLabel(decode_pred(preds))
        if decode_pred(preds) == 'Motorcycle':
            self.Motorcycle_guess.SetBackgroundColour((0,255,0))
        else:
            self.Motorcycle_guess.SetBackgroundColour((255,0,0))
            
        preds = model.predict(img_array_person)
        self.Person_guess.SetLabel(decode_pred(preds))
        if decode_pred(preds) == 'Person':
            self.Person_guess.SetBackgroundColour((0,255,0))
        else:
            self.Person_guess.SetBackgroundColour((255,0,0))
        
        
        self.Car_guess.SetLabel('Car')
        self.Car_guess.SetBackgroundColour((0,255,0))
        self.Motorcycle_guess.SetLabel('Motorcycle')
        self.Motorcycle_guess.SetBackgroundColour((0,255,0))
        self.Person_guess.SetLabel('Person')
        self.Person_guess.SetBackgroundColour((0,255,0))
        
        # Generate class activation heatmap
        heatmap_bus = make_gradcam_heatmap(img_array_bus, model, last_conv_layer_name)
        heatmap_car = make_gradcam_heatmap(img_array_car, model, last_conv_layer_name)
        heatmap_motorcycle = make_gradcam_heatmap(img_array_motorcycles, model, last_conv_layer_name)
        heatmap_person = make_gradcam_heatmap(img_array_person, model, last_conv_layer_name)
        
        
        def save_and_display_gradcam(img_path, heatmap, cam_path="output.jpg", alpha=0.4):
            # Load the original image
            img = keras.preprocessing.image.load_img(img_path)
            img = keras.preprocessing.image.img_to_array(img)
        
            # Rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)
        
            # Use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")
        
            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]
        
            # Create an image with RGB colorized heatmap
            jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
            jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        
            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * alpha + img
            superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        
            superimposed_img.save(cam_path)
        
        
        
        save_and_display_gradcam(im_bus, heatmap_bus, cam_path = 'E:/Brendan_2023_24/dump_tmp/bus_hm.png')
        save_and_display_gradcam(im_car, heatmap_car, cam_path = 'E:/Brendan_2023_24/dump_tmp/car_hm.png')
        save_and_display_gradcam(im_motorcycle, heatmap_motorcycle, cam_path = 'E:/Brendan_2023_24/dump_tmp/motorcycle_hm.png')
        save_and_display_gradcam(im_person, heatmap_person, cam_path = 'E:/Brendan_2023_24/dump_tmp/person_hm.png')
        

def main():
    app = wx.App()
    wind = EyesightGUI(None)
    wind.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()
