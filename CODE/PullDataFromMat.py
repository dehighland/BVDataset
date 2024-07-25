base_dir = "./Data/All_Images/"
final_csv_dir = base_dir + "Labels.csv"
dataframe_elements = [[],[],[],[],[],[], []]

for filename in os.listdir(data_dir):
	if "Patient " in filename:
		tempstr = data_dir + filename
		tempmat = scipy.io.loadmat(tempstr)
		
		pH = tempmat["pH"].item()
		apH = float('nan')
		Whiff = tempmat["Whiff"].item()
		Molecular = tempmat["Molecular"].item()
		Diagnosis = tempmat["Diagnosis"].item()
		
		clues = tempmat["ClueCells"][0]
	
		for idx in range(len(clues)):
			img = Image.fromarray(tempmat["Images"][0][idx], 'RGB')
			img_dir = base_dir + "pt" + filename.split(" ")[-1] + " (" + str(idx) +").png"
			img.save(img_dir, "PNG")
			
			dataframe_elements[0].append(img_dir)
			dataframe_elements[1].append(clues[idx])
			dataframe_elements[2].append(pH)
			dataframe_elements[3].append(Whiff)
			dataframe_elements[4].append(Molecular)
			dataframe_elements[5].append(Diagnosis)
			dataframe_elements[6].append(apH)
	
df = pd.DataFrame({"Image_Path": dataframe_elements[0], "pH": dataframe_elements[2], "ClueCell": dataframe_elements[1],
				   "Molecular": dataframe_elements[4], "Adj_pH": dataframe_elements[6], "Whiff": dataframe_elements[3],
				   "Diagnosis": dataframe_elements[5]})
				   
df.to_csv(final_csv_dir, index=False)