///////////////////////////////////////////////////////////////////////////
// Bayes-i modellezés a gyakorlatban.
// Tejelõ tehénállományok állományon belüli PTBC fertõzöttségének becslése  
//
// Veres Katalin, Lang Zsolt, Monostori Attila, Ózsvári László 
// 2024                                                       
///////////////////////////////////////////////////////////////////////////
// STAN MODELL
//
// Bemenõ adatok:  tehenek életkora, paritás, PTBC teszt eredmény 
// Kimenõ adatok: .... Bayesi becslése
///////////////////////////////////////////////////////////////////////////
// Használat: 
// Mentsük a fájlt a PTBC_egytelep fájlal azonos mappába.
// Ezt a fájl nem szükséges módosítani, sem közvetlenül futattni.
// A PTBC_egytelep fájl futtatásakor automatikusan beolvasásra kerül.
//////////////////////////////////////////////////////////////////////////

functions {  //inverze-béta eloszlásfüggvény
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
  int<lower=1> nAllRec;          // Az összes tehén korcsoportjainak száma
  int<lower=0> nAgegr_primi;     // Egyszer ellett tehenek korcsoportjainak száma
  int<lower=0> nCOW[nAllRec];    // Egyszer ill. többször ellett tehenek száma korcsoportonként 
  int<lower=0> POS[nAllRec];     // Pozitívat tesztelõ egyszer ill. többször ellett tehenek száma korcsoportonként
  real<lower=0> AgeG[nAllRec];   // A tehenek korcsoportjai
}
transformed data { //algebra-solverben használt paraméterek
  real x_r[0];
  int x_i[0];
  real rel_tol    = 10^(-4); //relatív hiba
  real f_tol      = 1;       //abszolút hiba
  real max_steps  = 10^4;    //iterációk maximális száma
}
parameters {
  real <lower = 0.00001, upper = 1> mu1;  // Egyszer ellett tehenekhez tartozó átlagos CWHP
  real <lower = 0.00001, upper = 1> mu2;  // Többször ellett tehenekhez tartozó átlagos CWHP
  real eta; // Telepszintû random hatás
  real et1; // Egyszer ellett tehenekhez tartozó additív random hatás
  real et2; // Többször ellett tehenekhez tartozó additív random hatás
  real <lower = 0.00001> sigmasq; // Telepszintû random hatás varianciája
  real <lower = 0.00001> sigm1sq; // additív random hatás varianciája - egyszer ellett tehenek
  real <lower = 0.00001> sigm2sq; // additív random hatás varianciája - többször ellett tehenek
  }

transformed parameters {
}

model {

  vector[3] theta; //inverz-béta eloszlásfüggvény bemenõ paramétereinek vektora
  
  real HTP;
  
  real lmu1;
  real lmu2;
  
  real CWHP1; // Egyszer ellett tehenekhez tartozó CWHP
  real CWHP2; // Többször ellett tehenekhez tartozó CWHP

  real pi1;    // Egyszer ellett tehenekhez tartozó látszólagos prevalencia
  real pi2;    // Többször ellett tehenekhez tartozó látszólagos prevalencia
  real Sp;     // Specificitás (fajlagosság)
  real Se;     // Korfüggõ szenzitivitás (érzékenység)
  real t;      // Loglikelihood függvény komponens

  real parA;
  real parB;
  real parC;

  real pszi1;
  real pszi2;

  real wgt1;
  real wgt2;
  real wgt3;
  real wgt4;
  
  // Korfüggõ szenzitivitást megadó kérlet paraméterei (Meyer et al. 2018)
  parA = 1.2; 
  parB = 3.0;
  parC = 0.30;

  // Specificitás (Meyer et al. 2018)
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
  
  //Diszperziós modell 
  pszi1 = (sigmasq+sigm1sq)^(-1); 
  pszi2 = (sigmasq+sigm2sq)^(-1);

  //varianciák súlyozása, súlyok négyzet összege 1
  wgt1  = sqrt(sigmasq/(sigmasq+sigm1sq));
  wgt2  = sqrt(sigm1sq/(sigmasq+sigm1sq));
  wgt3  = sqrt(sigmasq/(sigmasq+sigm2sq));
  wgt4  = sqrt(sigm2sq/(sigmasq+sigm2sq));
  
  theta[1] = mu1*pszi1;      // inverz-béta eloszlás "a" paramétere
  theta[2] = (1-mu1)*pszi1;  // inverz-béta eloszlás "b" paramétere
  theta[3] = (normal_cdf(wgt1*eta+wgt2*et1, 0, 1)+0.001)*0.999;
  CWHP1 = inv_logit(algebra_solver(inv_beta, [lmu1]', theta, x_r, x_i,
             rel_tol, f_tol, max_steps)[1]); 

  theta[1] = mu2*pszi2;
  theta[2] = (1-mu2)*pszi2;
  theta[3] = (normal_cdf(wgt3*eta+wgt4*et2, 0, 1)+0.001)*0.999;
  CWHP2 = inv_logit(algebra_solver(inv_beta, [lmu2]', theta, x_r, x_i,
             rel_tol, f_tol, max_steps)[1]);

    // Loglikelihood függvény komponens
    t = 0;
 
    // Egyszer ellett tehenek
    for (k in 1:nAgegr_primi) {

      // A k-adik korcsoporthoz tartozó szenzitivitás
      Se = inv_logit(parA-parB*exp(-parC*AgeG[k]));

      // Látszólagos prevalencia
      pi1   = Se*CWHP1 + (1-Sp)*(1-CWHP1);
      
      t += binomial_lpmf( POS[k] | nCOW[k], pi1);
    }

    // Többször ellett tehenek
    for (k in (nAgegr_primi+1):nAllRec) {

      // A k-adik korcsoporthoz tartozó szenzitivitás
      Se = inv_logit(parA-parB*exp(-parC*AgeG[k]));

      // Látszólagos prevalencia
      pi2  = Se*CWHP2 + (1-Sp)*(1-CWHP2);

      t += binomial_lpmf( POS[k] | nCOW[k], pi2);
    }
     // Apparent prevalence

    // Teljes loglikelihood
    target += t;
  
}

generated quantities { //egyedi cWHP számítása

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
