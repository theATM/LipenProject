def main():
    with open("out_LipenLabel.csv","w",encoding='utf-8') as out_file:
        with open("LipenLabel.csv","r",encoding='utf-8') as file:
            for line in file:
                name = line.split(";")[0]
                if name == "\n": continue
                tag = line.split(";")[1]
                subclass = line.split(";")[2]
                extra = line.split(";")[3]
                author = line.split(";")[4]
                if author[-1] != "\n":
                    ending = line.split(";")[5]
                else:
                    author = author[:-1]
                    ending = "\n"
                match tag:
                    case "Label":
                        pass
                    case "0":
                        subclass = str(list(range(1,-1,-1))[int(subclass)])
                    case "1" | "2":
                        pass
                    case "3":
                        subclass = str(list(range(3,-1,-1))[int(subclass)])
                    case "4":
                        subclass = str(list(range(2,-1,-1))[int(subclass)])
                    case "5":
                        subclass = str(list(range(1,-1,-1))[int(subclass)])
                    case "6":
                        pass
                newline = ";".join([name,tag,subclass,extra,author,ending])
                out_file.write(newline)






if __name__ == '__main__':
    main()