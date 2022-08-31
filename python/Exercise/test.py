def word_search(doc_list, keyword):
    final_list =[]
    for x in doc_list:
        val_list = x.casefold().split()
        val_list = [x.rstrip('.,') for x in val_list]
        if keyword.casefold() in val_list:
            final_list.append(doc_list.index(x))
    return final_list

print(word_search(['The Learn car Python Challenge Casino', 'They bought a car, and a horse'],'car'))
    