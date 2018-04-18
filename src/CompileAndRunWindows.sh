#!/bin/sh
javac -cp ".;./ejml-core-0.32.jar;./ejml-ddense-0.32.jar;./ejml-simple-0.32.jar" *.java
java -cp ".;./ejml-core-0.32.jar;./ejml-ddense-0.32.jar;./ejml-simple-0.32.jar" Faces "$@"
read -p "Press [Enter] key to exit..."