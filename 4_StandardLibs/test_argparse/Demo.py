#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

#argparse is a module that allows you to parse command line arguments
#example:

# create a parser object
parser = argparse.ArgumentParser(description='Process some integers.')
#1. add_argument() method is used to specify which command-line options the program is willing to accept
# it contains several arguments:
# name or flags - Either a name or a list of option strings, e.g. foo or -f, --foo.
# action - The basic type of action to be taken when this argument is encountered at the command line.
#     store: This is the default action, and stores the value argument to the destination.
#     store_const: This stores the value specified by the const keyword argument. The const keyword argument defaults to None.
#     store_true, store_false: These are special cases of 'store_const' for storing the values True and False respectively.
#     append: This stores a list, and appends each argument value to the list.
#     append_const: This stores a list, and appends the value specified by the const keyword argument to the list.
#     count: This counts the number of times a keyword argument occurs.
#     help: This prints a complete help message for all the options in the current command line and then exits.
#     version: This expects a version= keyword argument in the add_argument() call, and prints version information and exits.
# nargs - The number of command-line arguments that should be consumed. By default, one argument will be consumed and a single value will be produced.
#     Other values include:
#         N (an integer) consumes N arguments (and produces a list)
#         ? consumes zero or one arguments
#         * consumes zero or more arguments
#         + consumes one or more arguments
#         argparse.REMAINDER consumes all the remaining command-line arguments
# const - A constant value required by some action and nargs selections.
#    It could be a list of values, a single value, or a string, or a function that returns one of those types.
# default - The value produced if the argument is absent from the command line.
# type - The type to which the command-line argument should be converted.
# choices - A container of the allowable values for the argument.
# required - Whether or not the command-line option may be omitted (optionals only).
# help - A brief description of what the argument does.
# metavar - A name for the argument in usage messages.
# dest - The name of the attribute to be added to the object returned by parse_args().
# for example: 
#1. I want to add an argument that will take an integer value
parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
parser.add_argument('--a', metavar='N2', default= [0], type=int, nargs='*', help='another integer for the accumulator')
#2. add a flag that will sum the integers
parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, help='sum the integers')
#3. add a flag that will judge if sum will be executed
parser.add_argument('--ifsum', dest='if_sum', action='store_true', help='sum the integers')



#2. parse_args() method is used to parse the command line arguments
# it returns a Namespace object which contains the arguments to the command line
# it contains several arguments:
#   args - A list of strings to parse. The default is taken from sys.argv.
#   namespace - An object to take the attributes. The default is a new empty Namespace object.

args = parser.parse_args()
print(args.integers)
print(args.a)
if args.if_sum:
    print(args.accumulate(args.integers + args.a))

#we can run this script with the following command:
# python Demo.py 1 2 3 4
# python Demo.py 1 2 3 4 --sum
# python Demo.py 1 2 3 4 --sum 5 6 7 8
#what if I want --sum as an obliged flag?
#I can change the action to 'store_true' and add 'required=True'
# can we reorder the script inputs?
#like: python Demo.py --sum 1 2 3 4?
#yes, we can. argparse will parse the arguments and store them in the Namespace object
#what will be regarded as the integers in this case?
#the integers will be 1 2 3 4, and the sum will be the function sum
