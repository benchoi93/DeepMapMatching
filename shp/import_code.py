import os 
from qgis.core import QgsVectorLayer , QgsProject
os.chdir("D:/test_1/shp_file_2")
outdir_csv = "filtered_csv_2"
files = os.listdir(outdir_csv)

for i in range(30):
    filename=files[i]
    uri = "file:///D:/test_1/shp_file_2/"+outdir_csv+"/"+filename+"?delimiter={}&crs=epsg:4737&xField={}&yField={}".format(",", "x_coord", "y_coord")
    vlayer = QgsVectorLayer(uri, filename, "delimitedtext")
    QgsProject.instance().addMapLayer(vlayer)