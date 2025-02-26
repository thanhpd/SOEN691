fix bug from systems without any legacy init scripts
feature: add support for `Iop_Sar<width>` arithmetic shift right operator
Fix for #<I> Reacter process get exited with wrong sls & fixed build failed
Doc version bump\n\n[ci skip]
Memoize some (relatively expensive) calls in the Config class
Fix error for refactored static method.
Switch to using py_compile directly\n\nAlso allow specifying directories from command line.
Bugfix\n\n* fixed bug when no plugin was specified
no longer needlessly processing api output
Updated the asm code runner to free memory when done with it.
Add two more tests (setting __factory__ to None and deleting it).
Allow complete disabling of the member cache.
adding a __main__ block for testing the matching regex
fixed setup.py since no <I> was released, and my username has changed
removed long-deprecated seq method
Remove six from install_requires
Add support for Python <I>\n\nIn <I> codecs.lookup() started to normalize the encoding name, so there is a need to check for translit_type instead of translit/type
Adds Synthetic-Long-Read to strategy.
protect emcee3 between try/except
increase code coverage of SnapshotRetriever
Add missing class from PyQt4.QtGui
fix broken test after making change to input protocol
Lint the code - pylint seems to have trouble supporting type hints.
Pet peeve :-)
Bump vers for Latin synonyms\n\nFrom James Gawley. Contains Latin->Greek translations, too. More at <<URL>
fixed #<I> be adding support for __div__ in prefixedunit
Small bugfix in repelling from axes
Use the same name for the namedtuple used in the ErrorLogMonitor class.
Extra plotting defaults to lineiterator to simplify plotting code\n\nAdded plotforce to force plotting of an indicator which relies on non-plotted/plottable data/clock sources
Kill sleeping connections too as part of the failover process on the original master
look great\n\n...except for the changes to .onClose() -- This is to be overwritten by the developer so we should not include necessary code there.
Version bump to <I>.
change to $HOME during startup (#<I>)
Checks for variable names with the same name as Swift type keywords.\n\nIn the event that a spec variable has the name of a Swift keyword,\nan underscore is appended to the variable name.
Generate version classifiers dynamically in setup()
Remove obsolete ListView-based code\n\nI haven't replaced it, I just don't want it here!
fix(dsp): are_in_nested_dicts return false if nested_dict is not a dict.
Change version number to <I>.
Fix grammar mistake in debug log output
save etag after create instead of reloading\n\netag response header is in <I> POST responses\nsave this for future updates taking care not to clobber\nupdate response etags as well.
Exscriptd.RestClient: fix: error after successful placement of an order.
Added DerivedData class to DerivedTopic.
Change the default MD flavor from markdown_strict to markdown.
Add back missing superuser login in tests that was lost in merge
Added Pickle to __all__ declaration
Need to refactor this from the deparse, pos tagger, and elsewhere
Add: new contact parameter to know if the user is an expert or not. For UI usages
Less chatty plugin resolution logging (#<I>)\n\nWe were building a plugin Pex even if you have no plugins. When you do have plugins, it showed the internal type `PexRequirements(['foo'])`.\n\n\n\n[ci skip-rust]
quick fix SummarizationPipeline error messages (#<I>)\n\n* quick fix SummarizationPipeline error messages\n\n\n\nFix error messages to avoid spam errors, and errors of type:\n\n`Your max_length is set to <I>, but you input_length is only <I>. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=<I>)`\n\n\n\n* correcto SummarizationPipeline error messages fixes
Add field number_priority to powerpack api
added 'shared' parameter as discussed today with Andre.
Gaphas: Set port as focused_item\n\nIf only a single port is selected, it is set as focused_item (in\naddition to its selection)
Added all definition for painter package
- Improved multiprocess file locking\n- Added more out-of-core options to keep memory usage lower
fix: misuse of 'usage' command register parameter.\n\nthis corrects an unintentionally used 'usage' parameter to the intended 'help' parameter while decorating seach modes\nrefine and sort commands
Added forgotten req argument to a redirect_to_url() call.
mcnp metric rename
Altered the docstring for pymouse init
Fix runner.py to handle errors (#<I>)
fix: signBytes for impersonated credentials (#<I>)\n\n* fix: signBytes doesn't work for impersonated credentials\n\n\n\nFixes #<I>\n\n\n\n* black
IPython Console: Simplify some code
Return the package associated with a bz update.
[git enrich] hash field changed to not analyzed.
Bugfix catch NotADirectoryError alongside FileNotFoundError\n\nThis should prevent Hypercorn crashing if it tries to stat a file that\nis within a path that is not a directory, e.g. an egg binary file.
Fix bug in serialization of logging.LogRecord to show formatted message.
Change push subscription to delete subscription if it is no longer valid
Fixing typo on softlayer (again, so embarassing)
Add targetname property to Entity class
Warn if samesite is set but not supported\n\nThis is an adjustment to b<I>fc<I>bd<I>dd0df<I>c<I>bb<I>bd8d2f<I>c3 to\nensure that if samesite cookies are not supported there is a warning\nprinted. Samesite cookies are supported in Python >= <I>.
Temp fix for usage return field length
pprofile.zope: Fix _iterOutFiles signature.\n\nFixes a regression introduced in 5a3e8ffacaad<I>d<I>f<I>fb<I>e0cbfef<I>d2d6fb:\n  pprofile: Move cachegrind archive generation methods from pprofile.zope .
Fix path for images and annotations\n\nUpdate path for images and annotations\n\n\n\n1. From `<URL>
<I> Fixed readonly field during form state bound
Editor: Don't try to sync symbols and folding after the file is closed
Fix dependency versions in setup.py
[Seq2Seq] Correct import in Seq2Seq Trainer (#<I>)
Added shape property to NdElement
OAIRepository: CFG_OAI_REPOSITORY_MARCXML_SIZE\n\n* Increases CFG_OAI_REPOSITORY_MARCXML_SIZE from <I> to\n  <I> in order to reduce the overhead due to bibupload\n  creation.
fix pid output for SSH from Windows
Be sure to close hardware sources at ends of tests.\n\nsvn r<I>
Remove debug that can crash qemu\n\nFix #<I>
Add explicit encoding in title print to fix crash
Prefer relative imports in the django_auth_ldap package\n\nSlightly clearer and friendlier to renaming.
Added get and keys methods to AttrTree
Filter backup-results by given configuration.
Added get_remote_url method to Versioned mixin
Added check to announce function to check whether there is a message
Remove empty keys from compositor.info
fix overwriting python builtin
remove transform support from ScoreDecreaseFeatureImportances\n\nIt seems to be deprecated for sklearn estimators.
Changing AWS::Synthetics::Canary props, per May <I>, <I> update
docs: autodoc configuration and HTTPS fix\n\n* Replaces HTTP links with HTTPS where possible.
Fix key error with gateway, fixes #<I>
Use non-socket version of InteractiveExecutor for sos_run in sos-notebook
Change source API from pybel to bel
discoveryplus: fix an issue with some videos
propagate opts to salt.util.http call (#<I>)
bumped version to <I>
test search with where arg (breaks python>=<I>)
