#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

#argparse is a module that allows you to parse command line arguments

parser = argparse.ArgumentParser(description='This is a simple program to add two numbers')
parser.add_argument('--n1','--num1', help='The first number to add')
parser.add_argument('--num2', action='store', help='The second number to add')
#default, 
args = parser.parse_args()
if args.num1 is None or args.num2 is None:
    print("You need to pass two numbers")
else:
    print(int(args.num1) + int(args.num2))
#python3 Demo.py --num1 1 --num2 4