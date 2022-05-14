def read_api(val):
    import requests
    response = requests.get('https://api.direct.playstation.com/commercewebservices/ps-direct-us/products/productList?fields=FULL&productCodes='+val)
    
    response.raise_for_status()
    # access JSOn content
    jsonResponse = response.json()
    url_val = iterate(jsonResponse) 
    return url_val

def iterate(jsonResponse): 
    for key, value in jsonResponse.items(): 
        if isinstance(value, dict): 
            iterate(value) 
            continue  
        if key=="products":
            for k, v in value[0].items():
                if k=="baseOptions":
                    for a,b in v[0].items():
                       # print("a:{},b:{}".format(a,b))
                        if a=="selected":
                            return b["url"]
                 
def adding_toexcel():                  
    import pandas as pd
    df = pd.read_excel("C:\\Users\\HP\\Documents\\Production_SKU_Media_Map.xlsx")
    index = 0
    for x in df["SKU"]:
        if read_api(str(x)) is not None:
            print(str(index)+"::"+str(x)+"::"+read_api(str(x)))
            final_url = "https://direct.playstation.com"+read_api(str(x))
            df.at[index,'PDP URL']=final_url
        index= index+1
  
    df.to_excel("C:\\Users\\HP\\Documents\\Production_SKU_Media_Map.xlsx")

adding_toexcel()
