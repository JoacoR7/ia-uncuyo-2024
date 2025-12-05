import csv

def csv_summary(csv_path, column):
    reader = csv.DictReader(open(csv_path, newline='', encoding="utf-8"))
    min_value = float('inf')
    max_value = float('-inf')
    wins = 0

    count = 0
    total = 0

    for i, row in enumerate(reader):
        current_value = row[column]
        current_value = row[column].replace("[", "").replace("]", "").replace(".", "")
        current_value = float(current_value)
        if current_value < min_value:
            min_value = current_value
        if current_value > max_value:
            max_value = current_value
        if current_value >= 630:
            wins += 1
        
        total += current_value
        count += 1
    
    medium = total / count

    return min_value, max_value, medium, wins
