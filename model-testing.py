from similarity_pipeline import find_similar_materials

# Specify the path to your data file
data_file = "/Users/aryanrajsaxena/Desktop/BarryC/data_analysis/data-files/Master Data - Part 1.xlsx"

# Find similar materials for a given material code
material_code = "YYW-PN-G300297-E15"
similar_materials = find_similar_materials(material_code, data_file)

# View results
print(similar_materials[['Material_Code', 'Material_Group', 'similarity', 'score', 'used_count','Legislation']])