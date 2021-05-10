// variable to hold read result
long h0=0,h1=0,h2=0,h3=0,h4=0;

/// Vector of variables
// array of pointers to the read variables
long* hArrayPt[] = {&h0,&h1,&h2,&h3,&h4};
// calculating size of array
unsigned int const hArraySize = sizeof(hArrayPt)/sizeof(*hArrayPt);
// calculating end address for looping
// pointer points to first element in array
//long** hArrayEnd = &hArrayPt[hArraySize-1];
// array of values that will hold the dereferenced values
long hArrayData[hArraySize] = {0};

/// Grid of variables
// arranging the same pointers into a 2x2 array
long* hGrid[2][2] = {{&h0,&h1},{&h2,&h3}};
// store the size of the array
unsigned int const hGridRows = sizeof(hGrid)/sizeof(*hGrid);
unsigned int const hGridCols = sizeof(hGrid[0])/sizeof(*hGrid[0]);
// calculate end address for safe looping
//long** hGridEnd = &hGrid[hGridRows-1][hGridCols-1];
// grid of actual values that will be written
long hGridData[hGridRows][hGridCols] = {0};

//// print functions
/// print 1D vector
// print pointer array
template<size_t sz>
void printVector(long *(&arr)[sz]);
template<size_t sz>
void printVector(long (&arr)[sz]);

// print 2D grid of uniform size
// templates are used allow any sized 2d uniform array
template<size_t rows,size_t cols>
void printGrid(long *(&arr)[rows][cols]);
// data array
template<size_t rows,size_t cols>
void printGrid(long (&arr)[rows][cols]);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(A0,INPUT);
  pinMode(A1,INPUT);
  pinMode(A2,INPUT);
  pinMode(A3,INPUT);
  pinMode(A4,INPUT);

  /*
  char gridSizeStr[20];
  sprintf(gridSizeStr,"Vector size: %d",hArraySize);
  Serial.println(gridSizeStr);
  sprintf(gridSizeStr,"Array size (rxc): %d x %d",hGridRows,hGridCols);
  Serial.println(gridSizeStr);

  sprintf(gridSizeStr,"Data vector size: %d",sizeof(hArrayData)/sizeof(*hArrayData));
  Serial.println(gridSizeStr);
  sprintf(gridSizeStr,"Data Array: %d x %d",sizeof(hGridData)/sizeof(*hGridData),sizeof(hGridData[0])/sizeof(*hGridData[0]));
  Serial.println(gridSizeStr);
  Serial.println();
  Serial.println("h0,h1,h2,h3,h4    ");
  */
  Serial.flush();
  //exit(0);
}

void loop() {
  // read in and update variables
  h0 = analogRead(A0);
  h1 = analogRead(A1);
  h2 = analogRead(A2);
  h3 = analogRead(A3);
  h4 = analogRead(A4);

  // update arrays
  for(unsigned int i = 0;i<hArraySize;++i)
  {
    // dereference pointer array to 
    hArrayData[i] = *hArrayPt[i];
  }
  printVector(hArrayData);

  /*
  for(unsigned int r=0;r<hGridRows;++r)
  {
    for(unsigned int c=0;c<hGridCols;++c)
    {
      hGridData[r][c] = *hGrid[r][c];
    }
  }
  printGrid(hGridData);*/
  //Serial.flush();
  delay(500);
}

// print pointer array
template<size_t sz>
void printVector(long *(&arr)[sz])
{
  for(size_t i=0;i<sizeof(arr)/sizeof(*arr);++i)
  {
    Serial.print(*arr[i]);
    Serial.print(" ");
  }
  Serial.println();
}

// print data array
template<size_t sz>
void printVector(long (&arr)[sz])
{
  for(size_t i=0;i<sz;++i)
  {
    Serial.print(arr[i]);
    Serial.print(" ");
  }
  //delay(1000);
  Serial.println();
}

template<size_t rows,size_t cols>
void printGrid(long *(&arr)[rows][cols])
{
  /// iterate over array using printVector to print rows
  // assuming 2D, uniform
  Serial.println("Grid pt");
  for(size_t r=0;r<rows;++r)
  {
    for(size_t c=0;c<cols;++c)
    {
      Serial.print(*arr[r][c]);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println();
}

template<size_t rows,size_t cols>
void printGrid(long (&arr)[rows][cols])
{
  /// iterate over array using printVector to print rows
  // assuming 2D, uniform 
  for(size_t r=0;r<rows;++r)
  {
    for(size_t c=0;c<cols;++c)
    {
      Serial.print(arr[r][c]);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println();
}
