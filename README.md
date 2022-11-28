<h1>REVERSE IMAGE SEARCH</h1>
<h2>The dataset</h2>
<p>To get the images needed for annotation we <b>web scraped</b> product pictures from <b>amazon</b></p>
<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris1.png">

<h2>Generate inference file</h2>
<p>
	<i>
		img_size =224

		model = ResNet50(weights='imagenet', include_top=False,input_shape=(img_size, img_size, 3),pooling='max')

		batch_size = 64
		root_dir = '101_ObjectCategories'

		img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

		datagen = img_gen.flow_from_directory(root_dir,
												target_size=(img_size, img_size),
												batch_size=batch_size,
												class_mode=None,
												shuffle=False)

		num_images = len(datagen.filenames)
		num_epochs = int(math.ceil(num_images / batch_size))

		feature_list = model.predict_generator(datagen, num_epochs,verbose = 1)
	</i>
</p>

<h2>Results:</h2>
<p>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris2.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris3.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris4.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris5.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris6.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris7.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris8.png"><br>
	<img src="https://github.com/vpNjit/Reverse_Visual_Search/blob/main/ris/ris9.png"><br>
</p>
