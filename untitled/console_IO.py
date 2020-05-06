
def choose_checksum():
    i = -1
    while i not in [1, 2]:
        print("1. - CheckSum as sum of all characters in block mod 256 \n"
              "2. - CheckSum as CRC")
        try:
            i = int(input())
        except ValueError:
            print("choose 1 or 2")
            pass
    if i == 1:
        return "algebraic"
    elif i == 2:
        return "CRC"
    raise ValueError


def choose_COM():
    i = -1
    while i not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print("Choose port: \n"
              "1 - COM1\n"
              "2 - COM2\n"
              "3 - COM3\n"
              "4 - COM4\n"
              "5 - COM5\n"
              "6 - COM6\n"
              "7 - COM7\n"
              "8 - COM8\n"
              "9 - COM9\n")
        try:
            i = int(input())
        except ValueError:
            print("choose an INT between 1 and 9")
            pass
    strReturn = "COM" + str(i)
    return strReturn


def suma_kontrolna_algebraiczna(block):
    global checksumType
    if checksumType == "algebraic":
        suma = 0
        for i in block:
            suma += i
        suma = suma % 256
        return suma
    elif checksumType == "CRC":
        print(type(block[0]))
        crc = crc16.crc16xmodem(block)
        return crc
        pass
    return 0


def choose_file():
    i = ""
    while True:
        print("type file name with extension")
        try:
            i = str(input())
        except ValueError:
            continue
        break
    return i

if __name__ == "__main__":
    liczba = int(input())
    for i in range(0, max(0, liczba)):
        print(i)