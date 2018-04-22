def main () :
    data = open('train.txt', 'r', encoding='utf-8').read()
    trainData = data.split('\n')
    answers = []
    # for i in range(len(trainData)) :
    for i in range(3000) :
        temp = []
        temp = trainData[i].split('\t')
        answers.append(temp[2])
    return answers
if __name__ == "__main__" :
    main()