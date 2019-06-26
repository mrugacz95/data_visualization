# Data visualization and decomposition methods

Scripts with own implementations of PCA, EVD, and MDS methods used for visualization.

# Visualization

I've tested results of various decomposition methods:
* Custom MDS
* Sklearn MDS
* Sklearn Isomap
* Sklearn tSNE
* Sklearn Locally Linear embedding 

I used Kaggle's "Weedle's Cave" dataset to visualize distances between Pokémons.
To get use of Pokémon types I've changed raw string types such as "Grass", "Poison" to strengths and weaknesses against all other types. In result for example water and fire types are close because Pokémons of this types are strong against fire and ground but weak against grass.

Visualisation shows that next evolutions of Pokémons are often near each other. Also "Mega" Pokémons are close in visualization even if there wasn't direct information about that in dataset, what is really fascinating.
   
### Result
 
![Pokémon](https://user-images.githubusercontent.com/12548284/60219091-9892e180-9871-11e9-8209-7b555e2476cc.jpg)


# Image compression

With SVD it is also possible to compress images.
After constructing U, T and Vt matrices such as:

![matrices sizes](http://www.sciweavers.org/upload/Tex2Img_1561585191/render.png)

Where each matrix has defined size.

```
A (m x n)
U (m x m)
T (m x n)
Vt (n x n)
```

We can take k rows or columns to compress data, result with sizes:

```
A (m x n)
U (m x k)
T (k x k) (diagonal matrix)
Vt (k x n)
```

Summarizing, if image has 3 channels we can compress it with:

![compressed size](http://www.sciweavers.org/upload/Tex2Img_1561585657/render.png)

What it is less than original size
 
![orginal size](http://www.sciweavers.org/upload/Tex2Img_1561585793/render.png)

Have in mind that there are sophisticated algorithms for image compression such as JPG which can do it better.

### Usage

```bash
usage: compress.py [-h] -f INPUT_FILE [-out OUTPUT_FILE]
                   [-svd {sklearn,custom,numpy}] [-k K]
compress.py: error: the following arguments are required: -f
```

### Results

![mountains](https://user-images.githubusercontent.com/12548284/60219624-6e422380-9873-11e9-8ef0-6d9654f2e6c2.jpg)

![mountains compressed](https://user-images.githubusercontent.com/12548284/60219579-4226a280-9873-11e9-82e6-f0548b544220.png)

# Sources

* https://www.cs.put.poznan.pl/ibladek/students/skaiwd/
* http://timbaumann.info/svd-image-compression-demo/
* https://www.kaggle.com/terminus7/Pokémon-challenge
* https://raw.githubusercontent.com/zonination/Pokémon-chart/master/chart.csv