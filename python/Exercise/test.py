def word_search(doc_list, keyword):
    final_list =[]
    for x in doc_list:
        val_list = x.casefold().split()
        val_list = [x.rstrip('.,') for x in val_list]
        if keyword.casefold() in val_list:
            final_list.append(doc_list.index(x))
    return final_list

word_search(['The Learn Python Challenge Casino', 'They bought a car, and a horse', 'Casinoville?'],'car')
    