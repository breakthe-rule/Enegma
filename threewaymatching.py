import json

def extract_line_items(file_path, document_name):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    line_items = json_data.get('LineItems', [])
    
    normalized_items = []
    for item in line_items:
        normalized_items.append({
            "Id": item.get("Id"),
            "Description": item.get("Description"),
            "Quantity": float(item.get("Quantity", 0).replace(',', '')),
            "Price": float(item.get("Price", 0).replace(',', '')),
            "Totalprice": float(item.get("Totalprice", 0).replace(',', '')),
            "Document": document_name
        })
    return normalized_items

def compare_line_items(items1, items2, items3):
    
    docs = {"PO", "GR", "Invoice"}
    all_descriptions = {item["Description"] for item in items1} | {item["Description"] for item in items2} | {item["Description"] for item in items3}
    
    discrepancies = {
        "missing_items": [],
        "quantity_mismatch": [],
        "price_mismatch": [],
        "totalprice_mismatch": []
    }

    # Create dictionaries for easy lookup
    items1_dict = {item["Description"]: item for item in items1}
    items2_dict = {item["Description"]: item for item in items2}
    items3_dict = {item["Description"]: item for item in items3}

    for description in all_descriptions:
        item1 = items1_dict.get(description)
        item2 = items2_dict.get(description)
        item3 = items3_dict.get(description)
        
        quantities = {}
        prices = {}
        totalprices = {}
        
        if item1:
            quantities[item1["Document"]] = item1["Quantity"]
            prices[item1["Document"]] = item1["Price"]
            totalprices[item1["Document"]] = item1["Totalprice"]
        if item2:
            quantities[item2["Document"]] = item2["Quantity"]
            prices[item2["Document"]] = item2["Price"]
            totalprices[item2["Document"]] = item2["Totalprice"]
        if item3:
            quantities[item3["Document"]] = item3["Quantity"]
            prices[item3["Document"]] = item3["Price"]
            totalprices[item3["Document"]] = item3["Totalprice"]
        
        if len(quantities) < 3:
            discrepancies["missing_items"].append({
                "Description": description,
                "Documents": list(docs - set(quantities.keys()))
            })
        if len(set(quantities.values())) > 1:
            discrepancies["quantity_mismatch"].append({
                "Description": description,
                "Quantities": quantities
            })
        if len(set(prices.values())) > 1:
            discrepancies["price_mismatch"].append({
                "Description": description,
                "Prices": prices
            })
        if len(set(totalprices.values())) > 1:
            discrepancies["totalprice_mismatch"].append({
                "Description": description,
                "Totalprices": totalprices
            })
    
    return discrepancies

items1 = extract_line_items("invoice_data.json", "PO")
items2 = extract_line_items("invoice_data2.json", "GR")
items3 = extract_line_items("invoice_data3.json", "Invoice")

discrepancies = compare_line_items(items1, items2, items3)
print(discrepancies)
with open("discrepancies.json", 'w') as file:
    json.dump(discrepancies, file, indent=4)
    
print("Discrepancies saved in JSON file!")