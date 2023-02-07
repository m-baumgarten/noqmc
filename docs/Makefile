# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?=
SPHINXBUILD    ?= sphinx-build
SPHINXAPIDOC   ?= sphinx-apidoc
SOURCEDIR      = source
BUILDDIR       = build
APIDIR         = $(SOURCEDIR)/apidoc
PKGDIR         = ../noqmc

#EXCLUDEPATTERN = '../qcmagic/interfaces/liqcm/lib/*'

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile api

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile api
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

api:
	if [ -e $(APIDIR) ] ; then rm -R $(APIDIR); fi
	@$(SPHINXAPIDOC) -o $(APIDIR) -e $(PKGDIR) $(EXCLUDEPATTERN)
