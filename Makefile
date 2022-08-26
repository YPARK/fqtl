
PKG := $(shell cat DESCRIPTION | awk '$$1 == "Package:" { print $$2 }')
VER := $(shell cat DESCRIPTION | awk '$$1 == "Version:" { print $$2 }')

SRC := $(wildcard src/*.cc)
HDR := $(wildcard src/*.hh)
MAN := $(wildcard man/*.Rd)

all: $(PKG)_$(VER).tar.gz

clean:
	rm -f $(PKG)_$(VER).tar.gz src/*.o src/*.so

$(PKG)_$(VER).tar.gz: $(SRC) $(HDR) $(RR) $(MAN)
	rm -f $@
	R -e "Rcpp::compileAttributes(verbose=TRUE)"
	R -e "roxygen2::roxygenize();"
	R CMD build .

check: $(PKG)_$(VER).tar.gz
	R CMD check $<

install: $(PKG)_$(VER).tar.gz
	R CMD INSTALL $< 

site:
	R -e "pkgdown::build_site()"
