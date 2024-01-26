///////////////////////////////////////////////////////////////////////////
// Bayes-i modellez�s a gyakorlatban.
// Tejel� teh�n�llom�nyok �llom�nyon bel�li PTBC fert�z�tts�g�nek becsl�se  
//
// Veres Katalin, Lang Zsolt, Monostori Attila, �zsv�ri L�szl� 
// 2024                                                       
///////////////////////////////////////////////////////////////////////////
// STAN MODELL
//
// Bemen� adatok:  tehenek �letkora, parit�s, PTBC teszt eredm�ny 
// Kimen� adatok: .... Bayesi becsl�se
///////////////////////////////////////////////////////////////////////////
// Haszn�lat: 
// Ments�k a f�jlt a PTBC_egytelep f�jlal azonos mapp�ba.
// Ezt a f�jl nem sz�ks�ges m�dos�tani, sem k�zvetlen�l futattni.
// A PTBC_egytelep f�jl futtat�sakor automatikusan beolvas�sra ker�l.
//////////////////////////////////////////////////////////////////////////

functions {  //inverze-b�ta eloszl�sf�ggv�ny
  vector inv_beta (vector y,
                   vector theta,
                   real[] x_r,
                   int[] x_i) {
    vector[1] f_y;
    f_y[1] = beta_cdf(inv_logit(y[1]),theta[1],theta[2])-theta[3];
    return f_y;
  }
}
data {
  int<lower=1> nAllRec;          // Az �sszes teh�n korcsoportjainak sz�ma
  int<lower=0> nAgegr_primi;     // Egyszer ellett tehenek korcsoportjainak sz�ma
  int<lower=0> nCOW[nAllRec];    // Egyszer ill. t�bbsz�r ellett tehenek sz�ma korcsoportonk�nt 
  int<lower=0> POS[nAllRec];     // Pozit�vat tesztel� egyszer ill. t�bbsz�r ellett tehenek sz�ma korcsoportonk�nt
  real<lower=0> AgeG[nAllRec];   // A tehenek korcsoportjai
}
transformed data { //algebra-solverben haszn�lt param�terek
  real x_r[0];
  int x_i[0];
  real rel_tol    = 10^(-4); //relat�v hiba
  real f_tol      = 1;       //abszol�t hiba
  real max_steps  = 10^4;    //iter�ci�k maxim�lis sz�ma
}
parameters {
  real <lower = 0.00001, upper = 1> mu1;  // Egyszer ellett tehenekhez tartoz� �tlagos CWHP
  real <lower = 0.00001, upper = 1> mu2;  // T�bbsz�r ellett tehenekhez tartoz� �tlagos CWHP
  real eta; // Telepszint� random hat�s
  real et1; // Egyszer ellett tehenekhez tartoz� addit�v random hat�s
  real et2; // T�bbsz�r ellett tehenekhez tartoz� addit�v random hat�s
  real <lower = 0.00001> sigmasq; // Telepszint� random hat�s varianci�ja
  real <lower = 0.00001> sigm1sq; // addit�v random hat�s varianci�ja - egyszer ellett tehenek
  real <lower = 0.00001> sigm2sq; // addit�v random hat�s varianci�ja - t�bbsz�r ellett tehenek
  }

transformed parameters {
}

model {

  vector[3] theta; //inverz-b�ta eloszl�sf�ggv�ny bemen� param�tereinek vektora
  
  real HTP;
  
  real lmu1;
  real lmu2;
  
  real CWHP1; // Egyszer ellett tehenekhez tartoz� CWHP
  real CWHP2; // T�bbsz�r ellett tehenekhez tartoz� CWHP

  real pi1;    // Egyszer ellett tehenekhez tartoz� l�tsz�lagos prevalencia
  real pi2;    // T�bbsz�r ellett tehenekhez tartoz� l�tsz�lagos prevalencia
  real Sp;     // Specificit�s (fajlagoss�g)
  real Se;     // Korf�gg� szenzitivit�s (�rz�kenys�g)
  real t;      // Loglikelihood f�ggv�ny komponens

  real parA;
  real parB;
  real parC;

  real pszi1;
  real pszi2;

  real wgt1;
  real wgt2;
  real wgt3;
  real wgt4;
  
  // Korf�gg� szenzitivit�st megad� k�rlet param�terei (Meyer et al. 2018)
  parA = 1.2; 
  parB = 3.0;
  parC = 0.30;

  // Specificit�s (Meyer et al. 2018)
  Sp   = 0.995;
  
  mu1 ~ beta(65.7,715.9);
  mu2 ~ beta(134.195,716.1808);  
 
  sigmasq ~ inv_gamma(37.03,4.62);
  sigm1sq ~ inv_gamma(5.33,0.07);  
  sigm2sq ~ inv_gamma(6.05,0.09);  

  eta   ~ normal(0, 1); 
  et1   ~ normal(0, 1); 
  et2   ~ normal(0, 1); 

  lmu1=logit(mu1);
  lmu2=logit(mu2);
  
  //Diszperzi�s modell 
  pszi1 = (sigmasq+sigm1sq)^(-1); 
  pszi2 = (sigmasq+sigm2sq)^(-1);

  //varianci�k s�lyoz�sa, s�lyok n�gyzet �sszege 1
  wgt1  = sqrt(sigmasq/(sigmasq+sigm1sq));
  wgt2  = sqrt(sigm1sq/(sigmasq+sigm1sq));
  wgt3  = sqrt(sigmasq/(sigmasq+sigm2sq));
  wgt4  = sqrt(sigm2sq/(sigmasq+sigm2sq));
  
  theta[1] = mu1*pszi1;      // inverz-b�ta eloszl�s "a" param�tere
  theta[2] = (1-mu1)*pszi1;  // inverz-b�ta eloszl�s "b" param�tere
  theta[3] = (normal_cdf(wgt1*eta+wgt2*et1, 0, 1)+0.001)*0.999;
  CWHP1 = inv_logit(algebra_solver(inv_beta, [lmu1]', theta, x_r, x_i,
             rel_tol, f_tol, max_steps)[1]); 

  theta[1] = mu2*pszi2;
  theta[2] = (1-mu2)*pszi2;
  theta[3] = (normal_cdf(wgt3*eta+wgt4*et2, 0, 1)+0.001)*0.999;
  CWHP2 = inv_logit(algebra_solver(inv_beta, [lmu2]', theta, x_r, x_i,
             rel_tol, f_tol, max_steps)[1]);

    // Loglikelihood f�ggv�ny komponens
    t = 0;
 
    // Egyszer ellett tehenek
    for (k in 1:nAgegr_primi) {

      // A k-adik korcsoporthoz tartoz� szenzitivit�s
      Se = inv_logit(parA-parB*exp(-parC*AgeG[k]));

      // L�tsz�lagos prevalencia
      pi1   = Se*CWHP1 + (1-Sp)*(1-CWHP1);
      
      t += binomial_lpmf( POS[k] | nCOW[k], pi1);
    }

    // T�bbsz�r ellett tehenek
    for (k in (nAgegr_primi+1):nAllRec) {

      // A k-adik korcsoporthoz tartoz� szenzitivit�s
      Se = inv_logit(parA-parB*exp(-parC*AgeG[k]));

      // L�tsz�lagos prevalencia
      pi2  = Se*CWHP2 + (1-Sp)*(1-CWHP2);

      t += binomial_lpmf( POS[k] | nCOW[k], pi2);
    }
     // Apparent prevalence

    // Teljes loglikelihood
    target += t;
  
}

generated quantities { //egyedi cWHP sz�m�t�sa

  real CWHP1;
  real CWHP2;
  
  vector[3] theta;
  
  real pszi1x;
  real pszi2x;
  
  real wgt1x;
  real wgt2x;
  real wgt3x;
  real wgt4x;
  
  pszi1x = (sigmasq+sigm1sq)^(-1);
  pszi2x = (sigmasq+sigm2sq)^(-1);
  
  wgt1x  = sqrt(sigmasq/(sigmasq+sigm1sq));
  wgt2x  = sqrt(sigm1sq/(sigmasq+sigm1sq));
  wgt3x  = sqrt(sigmasq/(sigmasq+sigm2sq));
  wgt4x  = sqrt(sigm2sq/(sigmasq+sigm2sq));
  
  theta[1] = mu1*pszi1x;
  theta[2] = (1-mu1)*pszi1x;
  theta[3] = (normal_cdf(wgt1x*eta+wgt2x*et1, 0, 1)+0.001)*0.999;
  
   CWHP1 = inv_logit(algebra_solver(inv_beta, [logit(mu1)]', theta, x_r, x_i,
               rel_tol, f_tol, max_steps)[1]); 
               
   theta[1] = mu2*pszi2x;
   theta[2] = (1-mu2)*pszi2x;
   theta[3] = (normal_cdf(wgt3x*eta+wgt4x*et2, 0, 1)+0.001)*0.999;
   
   CWHP2 = inv_logit(algebra_solver(inv_beta, [logit(mu2)]', theta, x_r, x_i,
               rel_tol, f_tol, max_steps)[1]);
               
}
