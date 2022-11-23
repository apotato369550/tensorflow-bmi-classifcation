import random
import csv
import io


def progress_bar(progress, total):
    percent = 100 * (progress/float(total))
    bar = '#' * int(percent / 4) + '-' * int((100 - int(percent)) / 4)
    print(f"\r[{bar}] {percent:.2f}% row {progress} out of {total}...", end="\r")

rows = 2000

categories = ["Underweight", "Healthy", "Overweight", "Obese", "Morbidly_Obese"]


print("--- DATAGEN.PY ---")
print("Generating Training data: ")

with io.open("data/train.csv", "w", newline="", encoding='utf-8') as train:
    train_writer = csv.writer(train)
    train_writer.writerow(["height_cm", "weight_kg", "bmi", "category"])
    for i in range(rows):

        # generate a random height in centimeters from 135 to 235 cm, normally distributed
        height_cm = round(random.uniform(135, 210), 2)

        # generate a random bmi index from 18.5 to 40, normally distributed
        bmi = round(random.uniform(18.5, 40), 2)

        # calculate weight given those two values
        weight_kg = round(bmi * pow(height_cm / 100, 2), 2)

        # translate bmi index number to bmi category as a string
        if bmi < 18.5:
            index = 0
        elif bmi >= 18.5 and bmi <= 24.9:
            index = 1
        elif bmi >= 25 and bmi <= 29.9:
            index = 2
        elif bmi >= 30 and bmi <= 34.9:
            index = 3
        elif bmi >= 35:
            index = 4 

        # add as a row in data
        train_writer.writerow([height_cm, weight_kg, bmi, index])

        progress_bar(i + 1, rows)

print("Generating Evaluation data: ")

with io.open("data/eval.csv", "w", newline="", encoding='utf-8') as eval:
    eval_writer = csv.writer(eval)
    eval_writer.writerow(["height_cm", "weight_kg", "bmi", "category"])
    for i in range(rows):

        # generate a random height in centimeters
        height_cm = round(random.uniform(135, 210), 2)

        bmi = round(random.uniform(10, 40), 2)

        # calculate weight given those two values
        weight_kg = round(bmi * pow(height_cm / 100, 2), 2)

        # translate bmi index number to bmi category as a string
        
        if bmi < 18.5:
            index = 0
        elif bmi >= 18.5 and bmi <= 24.9:
            index = 1
        elif bmi >= 25 and bmi <= 29.9:
            index = 2
        elif bmi >= 30 and bmi <= 34.9:
            index = 3
        elif bmi >= 35:
            index = 4 

        # add as a row in data
        eval_writer.writerow([height_cm, weight_kg, bmi, index])

        progress_bar(i + 1, rows)