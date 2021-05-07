#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;
using namespace thrust;

// #define DocNum 3
// #define Classes 2
// #define DocWords 3
// #define DocClass_0 1
// #define DocClass_1 2

#define DocNum 10
#define Classes 2
#define DocWords 20
#define DocClass_0 5
#define DocClass_1 5

__host__ int isin(host_vector<string> vocab, string f) // just check that is string f in vector vocab?
{

    // cout << "debug " << f << endl;
    if (!vocab.empty())
    {
        // cout << "vocab not empty" << endl;
        for (int i = 0; i < vocab.size(); i++)
        {

            if (vocab[i].compare(f) == 0)
            {

                return i;
            }
        }
    }

    return -1;
};

__host__ int isinclassify(host_vector<string> &vocab, string Word) {
    for (int i = 0; i < vocab.size(); i++) {
        if (vocab[i].compare(Word) == 0) {
            return i;
        }
    }
    return -1;
}

__host__ void translateDoc( host_vector<string> vocabList,host_vector<string> docs, int* docWord_arr) {
    
    int index = 0;
    for (int i = 0; i < docs.size(); i++) {
        stringstream ssin(docs[i]);

        string word;
        while (ssin >> word)
        {
            docWord_arr[index] = isin(vocabList, word);
            index++;
        }
    }
}

__host__ void getVocab(host_vector<string> &docList, host_vector<string> &vocabList) {

    for (int i = 0; i < docList.size(); i++) {
        stringstream ssin(docList[i]);

        string word;
        // printf("%s\n", word);
        while (ssin >> word) {
            if (isin(vocabList, word) == -1){
                vocabList.push_back(word);
            }
        }
    }
    // for (int i = 0; i < DocNum; i++)
    // {

    //     stringstream ssin(docList[i]);

    //     string word;
    //     printf("%s\n", word);
    //     while (ssin >> word)
    //     {

    //         if (isin(vocabList, word) == -1)
    //         {

    //             vocabList.push_back(word);
    //         }
    //     }
    // }
}

__global__ void term_ClassN(int * doc, int * termInClass, int nDoc) {
    int tid = threadIdx.x;

    // printf("this is from term_ClassN thread %d\n", tid);

    for (int j = 0; j < nDoc*DocWords; j++) {
        
        if (tid == doc[j]) {
            // printf("thread id %d and doc word is %d\n",tid, doc[j]);
            termInClass[tid] = termInClass[tid] + 1;
        }
    }

}

__global__ void find_posterior(int * termInClass, int * nDoc_class, double * posteriorProb) {
    int tid = threadIdx.x;

    double pos = ((termInClass[tid] + 1) * 1.0) / ((*nDoc_class + 2) * 1.0);
    // printf("this is thread %d and pos is %lf add arr index %d\n",tid,pos,tid * (*cur_class));
    posteriorProb[tid] = pos;
    
}

__host__ void translateDocClassify(host_vector<string> &vocabList, string doc, int* docWord_arr) {

    string eachword;
    string Word[DocWords];
  
    // making a string stream
    stringstream ssin(doc);
  
    // Read and print each word.
    int i = 0;
    while (ssin >> eachword) {
        // cout << eachword << endl;
        // Word.push_back(eachword);
        docWord_arr[i] = isinclassify(vocabList, eachword);
        // cout << eachword << endl;
        Word[i] = eachword;
        i++;
    }
}

__global__ void classifyperthread(int *d_in, int *d_out, int* docWord_arr, int sizeofVocab, double *d_posteriorProb_class0,
                                    double *d_posteriorProb_class1) {

    int id = threadIdx.x;

    bool donthave = true;
    for (int i = 0; i < Classes; i++) {
        for (int j = 0; j < DocWords; j++) {
            if (d_in[j] == id) {
                if (i == 0) {
                    d_posteriorProb_class0[id] = d_posteriorProb_class0[id] * d_posteriorProb_class0[id]; 
                    // printf("class %d posteriorProb = %f\n", i, d_posteriorProb_class0[id]);
                    donthave = false;
                    break;
                }
                else {
                    d_posteriorProb_class1[id] = d_posteriorProb_class0[id] * d_posteriorProb_class0[id]; 
                    // printf("class %d posteriorProb = %f\n", i, d_posteriorProb_class1[id]);
                    donthave = false;
                    break;
                }
            }
        }
        if (donthave) {
            if (i == 0) {
                d_posteriorProb_class0[id] = d_posteriorProb_class0[id] * (1-(d_posteriorProb_class0[id]));
                // printf("class %d posteriorProb = %f\n", i, d_posteriorProb_class0[id]);
            }
            else {
                d_posteriorProb_class1[id] = d_posteriorProb_class0[id] * (1-(d_posteriorProb_class0[id]));
                // printf("class %d posteriorProb = %f\n", i, d_posteriorProb_class1[id]);
            }
        }
    }
}

__host__ int findMax(double *priorProb, double *posteriorProb_class0, double *posteriorProb_class1, int sizeofVocab) {
    vector<double> prob;
    for (int i = 0; i < Classes; i++) {
        for (int j = 0; j <= sizeofVocab; j++) {
            if (j == 0) {
                if (i == 0) {
                    prob.push_back(posteriorProb_class0[j]);
                    // prob.push_back(priorProb[i] * posteriorProb_class0[j]);
                }
                else {
                    prob.push_back(posteriorProb_class1[j]);
                    // prob.push_back(priorProb[i] * posteriorProb_class1[j]);
                }
            }
            else if (j < sizeofVocab) {
                if (i == 0) {
                    prob[i] *= posteriorProb_class0[j];
                }
                else {
                    prob[i] *= posteriorProb_class1[j];
                }
            }
            else if (j == sizeofVocab) {
                if (i == 0) {
                    prob[i] = prob[i] * priorProb[i];
                }
                else {
                    prob[i] = prob[i] * priorProb[i];
                }
            }
        }
    }
    cout << prob[0] << " " << prob[1] << endl;
    if (prob[0] > prob[1])
        return 0;
    else
        return 1;
}

int main() {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    host_vector<string> c_0;
    host_vector<string> c_1;

    //Test dataset
    // c_0.push_back("animal dog bird");
    // c_1.push_back("item box ball");
    // c_1.push_back("bottle item bomb");
    // string doc = "item bottle bird";
    // string doc = "animal dog bird";

    c_0.push_back("eligator hosting server we have hosting that can serve you Just paid 20 dollars per month for hosting your web");
    c_0.push_back("explore our selection of local favorites with 0 dollars delivery fee for your first month 10 dollars order minimum terms");
    c_0.push_back("need graphic design help in just a few clicks you can scale your creative output by hiring our pro designer");
    c_0.push_back("so your business is up and running now what grow with a marketing crm that gets smarter as you go");
    c_0.push_back("start and grow your business with shopify turn what you love into what you sell try shopify for free today");
    // c_0.push_back("looking for new glasses answer a few quick questions and we will suggest some great looking frames for you free");


    c_1.push_back("today I feel like I want to sleep all day I just wanna lay in my bed and go sleep");
    c_1.push_back("this week is rainy everyday I have to take my umbrella everyday it make me annoy sometimes when I walk");
    c_1.push_back("I am so tired I just want to rest in my vacation time go see outside not sit in table");
    c_1.push_back("she go to market to buy some pills but when she went out she forgot her wallet at her home");
    c_1.push_back("I am so tired now so I want to go to bed because I feel like I am not ok");

    // string doc = "create your website for your business just 399 dollars per month you can create your beautiful website for your business";
    // string doc = "I think I will go to sleep so do not disturb me I so tired now leave me alone please";
    // string doc = " I am so tired now so I want to go to bed because I feel like I am not ok";
    // string doc = "I feel like I am so tired I want to sleep everyday when I sleep in my bed feel good";
    string doc = "I am so tired now so I want to go to bed because I feel like I am not ok";
    // ***class 0 is ads class 1 is not ads***

    host_vector<string> vocabList;
    double priorProb[Classes];

    priorProb[0] = ((DocClass_0 + 1) * 1.0) / (((DocClass_0 + DocClass_1) + 2) * 1.0);
    priorProb[1] = ((DocClass_1 + 1) * 1.0) / (((DocClass_0 + DocClass_1) + 2) * 1.0);

    getVocab(c_0, vocabList);
    getVocab(c_1, vocabList);

    int class_0_arr[DocClass_0*DocWords];
    int class_1_arr[DocClass_1*DocWords];

    int termInClass_0[DocNum*DocWords];
    int termInClass_1[DocNum*DocWords];

    for (int t = 0; t < DocNum*DocWords; t++) {    // set value in termInClass to 0 for count in function
        termInClass_0[t] = 0;
        termInClass_1[t] = 0;
    }

    translateDoc(vocabList, c_0, class_0_arr);
    translateDoc(vocabList, c_1, class_1_arr);

    // kernel ---------------------------------------------------
    int * d_doc_array, *d_termInClass_0,*d_termInClass_1 ;


    // class 0

    cudaMalloc((void **) &d_doc_array, DocClass_0*DocWords*sizeof(int));
    cudaMalloc((void **) &d_termInClass_0, DocNum*DocWords*sizeof(int));
            
    cudaMemcpy(d_doc_array, &class_0_arr, DocClass_0*DocWords*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_termInClass_0, &termInClass_0, DocNum*DocWords*sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);

    term_ClassN<<<1,DocNum*DocWords>>>(d_doc_array, d_termInClass_0,DocClass_0);

    cudaMemcpy(&termInClass_0, d_termInClass_0, DocNum*DocWords*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_doc_array);
    // cudaFree(d_termInClass_0);

    // ---------------

    // class 1

    cudaMalloc((void **) &d_doc_array, DocClass_1*DocWords*sizeof(int));
    cudaMalloc((void **) &d_termInClass_1, DocNum*DocWords*sizeof(int));
            
    cudaMemcpy(d_doc_array, &class_1_arr, DocClass_1*DocWords*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_termInClass_1, &termInClass_1, DocNum*DocWords*sizeof(int), cudaMemcpyHostToDevice);

    term_ClassN<<<1,DocNum*DocWords>>>(d_doc_array, d_termInClass_1,DocClass_1);

    cudaMemcpy(&termInClass_1, d_termInClass_1, DocNum*DocWords*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_doc_array);
    // cudaFree(d_termInClass_1);


    int * d_nDoc_class ;
    
    double * d_posteriorProb_class0, *d_posteriorProb_class1;

    double posteriorProb_class0[DocWords*DocNum];
    double posteriorProb_class1[DocWords*DocNum];

    // posteriorProb class 0 ---------------------

    int size_of_docClass = DocClass_0;

    cudaMalloc((void **) &d_termInClass_0, DocNum*DocWords*sizeof(int));
    cudaMalloc((void **) &d_nDoc_class, sizeof(int));
    cudaMalloc((void **) &d_posteriorProb_class0, (DocWords*DocNum)*sizeof(double));

    cudaMemcpy(d_termInClass_0, &termInClass_0, DocNum*DocWords*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nDoc_class, &size_of_docClass, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posteriorProb_class0, &posteriorProb_class0, (Classes*DocWords*DocNum)*sizeof(double), cudaMemcpyHostToDevice);

    find_posterior<<<1,DocNum*DocWords>>>(d_termInClass_0, d_nDoc_class, d_posteriorProb_class0);

    cudaMemcpy(&posteriorProb_class0, d_posteriorProb_class0, (DocWords*DocNum)*sizeof(double), cudaMemcpyDeviceToHost);

    // cudaFree(d_termInClass_0);
    cudaFree(d_nDoc_class);
    // cudaFree(d_posteriorProb_class0);
        
    // -------------------------------------------

    // cout << "----------" << endl;

    // class 1 -----------------------------------

    size_of_docClass = DocClass_1;

    cudaMalloc((void **) &d_termInClass_1, DocNum*DocWords*sizeof(int));
    cudaMalloc((void **) &d_nDoc_class, sizeof(int));
    cudaMalloc((void **) &d_posteriorProb_class1, (DocWords*DocNum)*sizeof(double));

    cudaMemcpy(d_termInClass_1, &termInClass_1, DocNum*DocWords*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nDoc_class, &size_of_docClass, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_posteriorProb_class1, &posteriorProb_class1, (DocWords*DocNum)*sizeof(double), cudaMemcpyHostToDevice);

    find_posterior<<<1,DocNum*DocWords>>>(d_termInClass_1, d_nDoc_class, d_posteriorProb_class1);

    cudaMemcpy(&posteriorProb_class1, d_posteriorProb_class1, (DocWords*DocNum)*sizeof(double), cudaMemcpyDeviceToHost);

    // cudaFree(d_termInClass_1);
    cudaFree(d_nDoc_class);
    // cudaFree(d_posteriorProb_class1);

    // --------------------------------------------

    // show value of priorProb and posteriorProb

    // cout << endl <<"This is priorProb" << endl << endl;

    // for (int pp = 0 ; pp < Classes; pp++) {
    //     cout << priorProb[pp] << endl;
    // }

    // cout << endl << "this is posteriorProb" << endl << endl;

    // cout << "Class 0" << endl << endl;
    
    // for (int p = 0; p < DocWords*DocNum; p++) {
    //     cout << posteriorProb_class0[p] << endl;
    // }

    // cout << endl << "Class 1" << endl << endl;

    // for (int pd = 0; pd < DocWords*DocNum; pd++) {
    //     cout << posteriorProb_class1[pd] << endl;
    // }

    int docWord_arr[DocWords];

    //translate
    translateDocClassify(vocabList, doc, docWord_arr);
    //then get docWord_arr

    int *d_in;
    int *d_out;
    
	int h_out[1];

	cudaMalloc((void**) &d_in, DocWords*sizeof(int));
	cudaMalloc((void**) &d_out, sizeof(int));
    cudaMemcpy(d_in, &docWord_arr, DocWords*sizeof(int), cudaMemcpyHostToDevice);

    classifyperthread<<<1, DocWords*DocNum>>>(d_in, d_out, docWord_arr, vocabList.size(), d_posteriorProb_class0, d_posteriorProb_class1);

    cudaMemcpy(&posteriorProb_class0, d_posteriorProb_class0, (DocWords*DocNum)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&posteriorProb_class1, d_posteriorProb_class1, (DocWords*DocNum)*sizeof(double), cudaMemcpyDeviceToHost);
    cout << "Class = " << findMax(priorProb, posteriorProb_class0, posteriorProb_class1, vocabList.size()) << endl;

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_termInClass_0);
    cudaFree(d_termInClass_1);

    cudaFree(d_termInClass_0);
    cudaFree(d_posteriorProb_class0);

    cudaFree(d_termInClass_1);
    cudaFree(d_posteriorProb_class1);

	cudaFree(d_in);
	cudaFree(d_out);

	printf("time used: %f\n", milliseconds);
	
	return -1;
}