library(ggplot2)
#library(reshape2)
library(RColorBrewer)
#cols <- c("blue", "orange", "grey20", "red", "green", "yellow") 
cols <- rev(brewer.pal(9, "Blues"))

data.dir = "../data/" 
output.dir = data.dir

main<-function() {
    #visualize.aucs.feature()
    visualize.aucs()
}

auc.difference<-function() {
    out.file = paste0(data.dir, "validation.dat")
    d = read.csv(out.file, sep="\t")
    e = d[d$n_fold==10 & d$n_proportion==2 & d$disjoint=="True" & d$features=="chemical|target|phenotype",]
    e = e[1:10,]
    f = d[d$n_fold==10 & d$n_proportion==2 & d$disjoint=="False" & d$features=="chemical|target|phenotype",]
    f = f[1:10,]
    a=t.test(as.numeric(as.character(e$auc.mean)), as.numeric(as.character(f$auc.mean)))
    print(a$p.value)
}


visualize.aucs<-function() {
    features = c("chemical|target|phenotype")
    out.file = paste0(data.dir, "validation_ddi.dat")
    d = read.csv(out.file, sep="\t")
    #e = d[d$features %in% features & d$variable == "avg" & d$n_fold==10 & d$n_proportion==2 & d$disjoint=="True",]
    e = d[d$features %in% features & d$variable == "avg",]
    e$n_subset = paste0(e$n_subset / 1000, "K")
    e$n_subset = factor(e$n_subset, levels=c("1K", "5K", "10K"))
    e$disjoint = ifelse(e$disjoint == "True", "Disjoint", "Non-disjoint")
    e$disjoint = factor(e$disjoint, levels=c("Non-disjoint", "Disjoint"))
    e$n_fold = ifelse(e$n_fold == "10", "10-fold", "2-fold")
    #e$n_fold = factor(e$n_fold, levels=c("10-fold", "2-fold"))
    print(e)
    #e$auc.sd = 2*e$auc.sd / sqrt(10) # std err 
    dodge = position_dodge(width=0.9)
    p = ggplot(data=e, aes(n_subset, auc.mean)) + geom_bar(aes(fill=n_subset), stat="identity", position=dodge, alpha=0.6) 
    p = p + geom_errorbar(data=e, aes(x = n_subset, ymax = (auc.mean + auc.sd), ymin = (auc.mean - auc.sd)), width=0.2)
    p = p + coord_cartesian(ylim=c(50,95))
    p = p + labs(x="Number of positive instances", y="AUC (%)") + facet_wrap(~disjoint + n_fold)
    p = p + theme_bw() 
    p = p + scale_fill_manual(values = cols)
    p = p + guides(fill=F) 
    txt.size = 18
    p = p + theme(text = element_text(size = txt.size), axis.text.x = element_text(size=txt.size, angle=0, vjust=0.5, hjust=0.5), axis.text.y = element_text(size = txt.size)) 
    p = p + theme(plot.background = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank()) # panel.grid.major = element_blank()
    p = p + theme(axis.ticks.x=element_blank(), axis.line = element_line(color = 'black'))
    #p = p + theme(legend.text = element_text(size=txt.size)) # legend.position=c(0.9, 0.9)
    out.file = paste0(output.dir, "auc.svg") 
    #pdf(out.file, bg="white") #!
    svg(out.file) 
    print(p)
    dev.off()
}


visualize.aucs.feature<-function() {
    features = c("chemical", "target", "phenotype", "chemical|target|phenotype")
    out.file = paste0(data.dir, "validation.dat")
    d = read.csv(out.file, sep="\t")
    e = d[d$features %in% features & d$variable == "avg" & d$n_fold==10 & d$n_proportion==2 & d$disjoint=="True",]
    print(e)
    e$auc.mean = round(as.numeric(as.character(e$auc.mean)), digits=1)
    e$auc.sd = round(as.numeric(as.character(e$auc.sd)), digits=1)
    #e$auc.sd = 2*auc.sd / sqrt(10) # std err 
    e$features = factor(e$features, levels=c(features, "side effect", "combined"))
    e[e$features == "phenotype", "features"] = "side effect"
    e[e$features == "chemical|target|phenotype", "features"] = "combined"
    e$features = factor(e$features, levels=c("chemical", "target", "side effect", "combined"))
    # auc.mean auc.sd
    p = ggplot(data = e, aes(x = features, y = auc.mean)) + geom_bar(width = 0.8, stat="identity", fill="darkorange") #, alpha=0.5) # aes(fill=features), 
    p = p + geom_errorbar(data=e, aes(x = features, ymax = (auc.mean + auc.sd), ymin = (auc.mean - auc.sd)), width=0.2)
    p = p + coord_cartesian(ylim=c(50,67))
    p = p + labs(x=NULL, y="AUC (%)") 
    p = p + theme_bw() 
    #p = p + guides(fill=F) 
    txt.size = 18
    p = p + theme(text = element_text(size = txt.size), axis.text.x = element_text(size=txt.size, angle=15, vjust=0.5, hjust=0.5), axis.text.y = element_text(size = txt.size)) 
    #p = p + theme(plot.background = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank()) # panel.grid.major = element_blank()
    p = p + theme(plot.background = element_blank(), panel.grid.minor = element_blank()) # axis.ticks.x=element_blank()
    p = p + theme(axis.ticks.x=element_blank(), axis.line = element_line(color = 'black'))
    #p = p + theme(legend.text = element_text(size=txt.size)) # legend.position=c(0.9, 0.9)
    out.file = paste0(output.dir, "features.pdf") 
    pdf(out.file, bg="white")
    print(p)
    dev.off()
}


main()

