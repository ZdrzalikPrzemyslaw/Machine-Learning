try:
    liczba1 = float(input("podaj pierwsza liczbe "))
    liczba2 = float(input("podaj druga liczbe "))
    liczba3 = float(input("podaj trzecia liczbe "))
    liczba4 = float(input("podaj czwarta liczbe "))
except ValueError:
    print("ValueError :(")
    exit(3)

srednia = (liczba1 + liczba2 + liczba3 + liczba4)/4
print("Srednia z 4 liczb = ", srednia)

keyPressed = " "
while keyPressed != "":
    keyPressed = ""
    keyPressed = str(input("Nacisnij enter albo ci zajebe"))