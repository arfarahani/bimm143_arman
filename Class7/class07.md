# Class 7: Machine Learning I
Arman Farahani A17497672

Today we are going to learn how to apply different machine learning
methods, begining with clustering:

The goal hee is to find groups/clusters in your input data.

First II will make up some data with clear groups. For tthis I will use
the `rnorm()` function.

``` r
rnorm(10)
```

     [1] -0.20303429 -0.33907828  0.28594348 -0.75320166 -0.55062339 -0.09432093
     [7]  0.80617950 -0.37404315  2.06854126  0.67064541

``` r
hist( rnorm(10000) )
```

![](class07_files/figure-commonmark/unnamed-chunk-2-1.png)

``` r
hist( rnorm(10000, mean=3)  )
```

![](class07_files/figure-commonmark/unnamed-chunk-3-1.png)

``` r
n <- 30
x <- c(rnorm(n, -3), rnorm(n, +3))
y <- rev(x)
y
```

     [1]  2.5402360  2.2297818  3.8081875  4.2712419  2.8979089  2.4951448
     [7]  1.0900489  3.4840752  3.5141283  2.6961132  1.7194625  3.0282249
    [13]  1.8463140  2.5345330  2.5534255  1.8615169  4.0469188  2.7481563
    [19]  5.6295153  1.8888039  3.8869884  0.8178791  3.7446220  1.8863068
    [25]  2.3639857  3.6321732  3.6778302  2.1811310  4.1152278  1.1425531
    [31] -1.8506012 -1.8049336 -2.7379812 -3.2239945 -2.8083053 -3.1905017
    [37] -3.1239201 -1.6310132 -3.6128727 -1.6730723 -3.4071248 -3.8548956
    [43] -3.4230155 -4.2278683 -2.9760558 -3.9685780 -1.9374845 -2.8353589
    [49] -3.4612250 -2.2844993 -4.1959060 -4.2621465 -2.3042405 -3.3127333
    [55] -3.9142262 -4.2859008 -2.4962288 -3.5658202 -4.0110186 -3.6189979

``` r
z <- cbind(x, y)
head(z)
```

                 x        y
    [1,] -3.618998 2.540236
    [2,] -4.011019 2.229782
    [3,] -3.565820 3.808188
    [4,] -2.496229 4.271242
    [5,] -4.285901 2.897909
    [6,] -3.914226 2.495145

``` r
plot(z)
```

![](class07_files/figure-commonmark/unnamed-chunk-5-1.png)

> Q. How many points are in each cluster?

> Q. What ‘component’ of your result objet details -cluster size
> -cluster assignment/membership -cluster center

> Q. Plot x colored by the kmeans cluster assignment and add cluster
> centers as blue points

``` r
km <- kmeans(z, centers = 2)
km
```

    K-means clustering with 2 clusters of sizes 30, 30

    Cluster means:
              x         y
    1  2.811081 -3.133351
    2 -3.133351  2.811081

    Clustering vector:
     [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1
    [39] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

    Within cluster sum of squares by cluster:
    [1] 54.38074 54.38074
     (between_SS / total_SS =  90.7 %)

    Available components:

    [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
    [6] "betweenss"    "size"         "iter"         "ifault"      

Results in kmeans object `km`

``` r
attributes(km)
```

    $names
    [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
    [6] "betweenss"    "size"         "iter"         "ifault"      

    $class
    [1] "kmeans"

cluster size?

``` r
km$size
```

    [1] 30 30

cluster assignment/membership?

``` r
km$cluster
```

     [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1
    [39] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

cluster center

``` r
km$center
```

              x         y
    1  2.811081 -3.133351
    2 -3.133351  2.811081

> Q. Plot x colored by the kmeans cluster assignment and add cluster
> centers as blue points

R will re-cycle the shorter color vector to be the same length as the
longer (number of data points) in z

``` r
plot(z, col=c("red", "blue"))
```

![](class07_files/figure-commonmark/unnamed-chunk-11-1.png)

``` r
plot(z, col=3)
```

![](class07_files/figure-commonmark/unnamed-chunk-12-1.png)

``` r
plot(z, col=km$cluster)
points(km$centers, col="blue", pch=19)
```

![](class07_files/figure-commonmark/unnamed-chunk-13-1.png)

> Q. Can you run kmeans and ask for 4 clusters please and plot the
> results like we have done above?

``` r
km1 <- kmeans(z, center=4)
plot(z, col=km1$cluster)
points(km1$centers, col="blue", pch=19)
```

![](class07_files/figure-commonmark/unnamed-chunk-14-1.png)

Hierarchical clustering

Let’s take our same data `z` and see how hclust works.

``` r
d <- dist(z)
hc <- hclust(d)
hc
```


    Call:
    hclust(d = d)

    Cluster method   : complete 
    Distance         : euclidean 
    Number of objects: 60 

``` r
plot(hc)
abline(h=8, col="red")
```

![](class07_files/figure-commonmark/unnamed-chunk-16-1.png)

I can get my cluster membership vector by “cutting the tree” with the
`cutree()` function like so:

``` r
grps <- cutree(hc, h=8)
grps
```

     [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2
    [39] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

Can you plot `z` colored by our hclust results:

``` r
plot(z, col=grps)
```

![](class07_files/figure-commonmark/unnamed-chunk-18-1.png)

## PCA of UK food data

Read data from the UK on food consumption in different parts of the UK

``` r
url <- "https://tinyurl.com/UK-foods"
  x <- read.csv(url, row.names=1)
  head(x)
```

                   England Wales Scotland N.Ireland
    Cheese             105   103      103        66
    Carcass_meat       245   227      242       267
    Other_meat         685   803      750       586
    Fish               147   160      122        93
    Fats_and_oils      193   235      184       209
    Sugars             156   175      147       139

``` r
barplot(as.matrix(x), beside=F, col=rainbow(nrow(x)))
```

![](class07_files/figure-commonmark/unnamed-chunk-20-1.png)

A so-called “Pairs” plot can be useful for small datasets like this one

``` r
pairs(x, col=rainbow(10), pch=16)
```

![](class07_files/figure-commonmark/unnamed-chunk-21-1.png)

It is hard to see structure and trends in even this small dataset. How
will we ever do this when we have big datasets with 1,000s or 10s of
thousands of things we are measuring…

### PCA to the rescue

Let’s see how PCA deals with this dataset. So main function in base R to
do PCA is called `prcomp()`

``` r
pca <- prcomp( t(x) )
summary(pca)
```

    Importance of components:
                                PC1      PC2      PC3       PC4
    Standard deviation     324.1502 212.7478 73.87622 3.176e-14
    Proportion of Variance   0.6744   0.2905  0.03503 0.000e+00
    Cumulative Proportion    0.6744   0.9650  1.00000 1.000e+00

Let’s see what is inside this `pca` object that we created from running
`prcomp()`

``` r
attributes(pca)
```

    $names
    [1] "sdev"     "rotation" "center"   "scale"    "x"       

    $class
    [1] "prcomp"

``` r
pca$x
```

                     PC1         PC2        PC3           PC4
    England   -144.99315   -2.532999 105.768945 -4.894696e-14
    Wales     -240.52915 -224.646925 -56.475555  5.700024e-13
    Scotland   -91.86934  286.081786 -44.415495 -7.460785e-13
    N.Ireland  477.39164  -58.901862  -4.877895  2.321303e-13

``` r
plot(pca$x[,1], pca$x[,2], col=c("black", "red", "blue", "green"), pch=16, xlab="PC1 (67.4%)", ylab= "PC2(29%)")
```

![](class07_files/figure-commonmark/unnamed-chunk-25-1.png)

## Quarto

Quarto enables you to weave together content and executable code into a
finished document. To learn more about Quarto see <https://quarto.org>.

## Running Code

When you click the **Render** button a document will be generated that
includes both content and the output of embedded code. You can embed
code like this:

``` r
1 + 1
```

    [1] 2

You can add options to executable code like this

    [1] 4

The `echo: false` option disables the printing of code (only output is
displayed).
