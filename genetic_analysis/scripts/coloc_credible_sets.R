#!/usr/bin/env Rscript
aa<-commandArgs(FALSE);script<-sub('^--file=','',aa[grepl('^--file=',aa)])
root<-normalizePath(file.path(dirname(script),'..'))
suppressPackageStartupMessages(library(data.table));suppressPackageStartupMessages(library(susieR));suppressPackageStartupMessages(library(coloc))
policies<-data.table(policy=c('coverage_95_min_ld_0.5','coverage_80_min_ld_0.5','coverage_95_min_ld_0.2'),
                    coverage=c(.95,.80,.95),min_abs_corr=c(.5,.5,.2))
counts<-list();sets<-list();pairs<-list()
for(window in c('100kb','1Mb')){
  reselected<-list()
  for(label in c('CD14_Mono_expression','CD16_Mono_expression','MS')){
    folder<-file.path(root,'work',paste0(window,'_IFNGR1_',label))
    fit<-readRDS(file.path(folder,'full_trait_susie_fit.rds'));h<-fread(file.path(folder,'statistics.tsv'));p<-nrow(h)
    con<-file(file.path(folder,'LD.bin'),'rb');v<-readBin(con,'double',n=p*p,size=8,endian='little');close(con)
    R<-matrix(v,nrow=p,byrow=TRUE);rm(v)
    for(j in seq_len(nrow(policies))){
      policy<-policies[j];set.seed(20261105)
      cs<-susie_get_cs(fit,Xcorr=R,coverage=policy$coverage,min_abs_corr=policy$min_abs_corr)
      counts[[length(counts)+1]]<-data.table(window=window,trait=label,policy=policy$policy,
        coverage=policy$coverage,min_abs_corr=policy$min_abs_corr,n_credible_sets=length(cs$cs),
        components=paste(cs$cs_index,collapse=';'),posterior_refitted=FALSE)
      if(j==1)stopifnot(setequal(cs$cs_index,fit$sets$cs_index))
      fresh<-fit;fresh$sets<-cs;reselected[[paste(label,policy$policy)]]<-fresh
      if(length(cs$cs))for(k in seq_along(cs$cs)){
        component<-cs$cs_index[k];top<-which.max(fit$alpha[component,])
        sets[[length(sets)+1]]<-data.table(window=window,trait=label,policy=policy$policy,
          component=component,n_variants=length(cs$cs[[k]]),min_abs_corr=cs$purity$min.abs.corr[k],
          component_log_BF=fit$lbf[component],max_alpha=fit$alpha[component,top],
          top_SNP_GRCh37=h$snp_GRCh37[top],claimed_coverage=cs$coverage[k])
      }
    }
    rm(R,fit,fresh);gc(FALSE)
  }
  for(policy in policies$policy)for(label in c('CD14_Mono_expression','CD16_Mono_expression')){
    d<-reselected[[paste('MS',policy)]];e<-reselected[[paste(label,policy)]]
    cc<-coloc.susie(d,e,p1=1e-4,p2=1e-4,p12=1e-5)
    if(is.list(cc)&&!is.null(cc$summary)&&nrow(cc$summary)){
      z<-as.data.table(cc$summary);z[,`:=`(window=window,cell_type=sub('_expression$','',label),policy=policy,posterior_refitted=FALSE)]
      pairs[[length(pairs)+1]]<-z
    }
  }
}
fwrite(rbindlist(counts,fill=TRUE),file.path(root,'results/coloc_credible_set_counts.csv'))
fwrite(rbindlist(sets,fill=TRUE),file.path(root,'results/coloc_credible_sets.csv'))
fwrite(rbindlist(pairs,fill=TRUE),file.path(root,'results/coloc_credible_set_sensitivity.csv'))
