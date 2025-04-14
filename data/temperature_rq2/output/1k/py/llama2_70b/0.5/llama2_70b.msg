Detached a buffer from the chunk adapter and disconnected the port from the NodeMap for device {0}.
Replace get_list with list_buckets in S3Manager
Added check for is_ec2_instance before reporting error
Added profiling for server loop using cProfile
Refactored Time class to use MetricUnit for handling intervals of time.
Update parser.py to support Python 2 and 3
Fix assertion in testStore.py
Update format string in FileTest to use {0} instead of {1} for consistency.
Removed unnecessary print statements in unescape function.
Added code to print traceback information when encountering ImportError while loading modules in MAVProxy.
Update version number to 0.2.0 and copyright year to 2021
Update pandas repository path and benchmark timespan.
Added option to disable cross-seeding and set additional label for cross-seeding in MetafileCreator.
MAGICCBase class updated to handle debug argument for MAGICC7 and above, and to raise a ValueError if the user attempts to use debug with MAGICC6.
Added a try-except block in Interpreter.value to handle ValueError for renorm and added a warning when force=True is used.
Remove unnecessary close() call in get_file_chunk_consumer() when handling a fifo file.
Added _MIN_LEN variable and updated register_bpdu_type method to include _MIN_LEN check.
Added code to enable reporting for input pins in Port class.
Fix for incorrect encoding in SortedHostCollection.pop() method.
Removed unnecessary 'legislative_input' parameter from Column constructor.
Added logging for vault information retrieval and handling of exceptions
Update version format in OBO parser
Added color values and reference link to powerline-bash.py
Modified Address class to use classmethod for b58decode and fixed parameter error handling.
Update doc_tree.py to handle missing YAML blocks
Update test_toolbox_html_tools.py to remove unnecessary print statement and fix assertions.
Update send_file function to use cache_timeout and last_modified parameters.
Replace _LOGGER.exception with _LOGGER.debug in heartbeater function.
Modified Method class in web3/method.py to correctly apply response formatters.
Added probe service and constants
Added parameter 'group' to list_metafeatures() and added a new group 'landmarking' to filter metafeatures by.
Updated regular expression for parsing CRTF files
Added check for _connection before refreshing schema and node list
Added check to prevent creating a directory inside itself in mkdir_p function.
Added support for releasing documentation with Sphinx and updated wheel dependencies.
Update version number to 1.0.2
Added darwin to list of supported platforms in platforms() function.
Added code to ensure client disconnection and reconnection in interactive function client1.
Bump version number to 4.0.2
Bump version number to 0.10.1.
Added test for index dot slices

Added data files for scripts and updated setup.py
Added flake8 directive to disable linting warnings in Django models.py file.
Set MACOSX_DEPLOYMENT_TARGET to 10.7 for OSX wheel builds.
Allow window_shell to be optional in Session constructor
Update wheel version from 0.29 to 0.30.
Update matplotlib.use('agg') to matplotlib.use('Agg') in syncpr_tests.py
Added a comment to the parse function in ttlfmt.py to indicate that we don't need to reset saved parsed streams to the point where they fail.
Added option to specify AWS profile and instance type
Added captureoutput tag for capturing template output into a variable
Disable close_on_teardown for PyMemcache in test_flask_pymemcache.py
Update version number to 1.4.0
Added blur option to TDHeatmap constructor in test_heat_map_withtime.py
Added test for code blocks and fixed syntax boundaries in CMark parser.
Added support for two new voices: Joanna (en-US) and Mizuki (ja-JP).
Added detection for Read the Docs environment and adjusted sys.path accordingly.
Added support for Dask-based NinjoTIFF writing
Modified GitFormula to reuse existing cloned repository instead of re-cloning.
Increased thumbnail width and height in image.py
Added logging for unexpected engine output in XBoardProtocol.
Added bool_only parameter to generic.py, allowing for selection of only boolean columns.
Update Action.to_rest to include dp as an argument
Fix for k8s tests failing due to missing images
Fixes #3475: Add test for deleting a domain without removing others
utils.py: Skipped read-only attribute fields when the resource is non-existent
Added root_url parameter to JSHost constructor in host.py
Modified path.py to use the new io.open function instead of the old open function.
Update Renderer to use partials instead of loader
Update conftest.py to properly handle homely modules and environment variables.
Replace OpenSSL.SSL.Connection with Connection
Handle property definitions in _impl phase
Add support for ellipses in ShapeList
Added test for empty response content in head requests
Fix Varscan VCF output: move temporary file to original file and update header with sample names from input BAMs.
Added support for passing configuration to message processor constructor.
Simplify boolean AND expressions by flattening them and checking if we are left with a single argument.
Added import of Board in Session class to ease circular import issues.
Expand exception types in _TemporalAttribute.parse_datetime to include TypeError.
Update generate_json_docs.py to use 'gcloud' instead of 'gcloud-python' in clean_source_path function.
Update tile_directory.py to comment out segmentize_on_clip parameter.
Added tests for check_request with allowed caps and describeprocess
Update resnet_fpn_backbone to use trainable_layers instead of freezing layers
Added exception for chopping datetime type in Calendar class.
Added test for reading empty arrays with various dtypes using pyarrow
Added permission checking for yAuth decorator
Added function get_token_features and updated gram_label and gender_match_label to use it.
Allow selecting and copying whole columns in ArrayView
Added test for fuse projections
Added new properties to StageDescription class: CacheDataEncrypted, CacheTtlInSeconds, CachingEnabled, DataTraceEnabled, LoggingLevel, MetricsEnabled, ThrottlingBurstLimit, and ThrottlingRateLimit.
Removed unnecessary prints in SqlaTable.order_by
Added shortcuts for calling methods on Lens objects
Removed source_parsers and updated source_suffix to only include .rst files.
Update wait_for_droplet_to_be_created to use droplet UIDs instead of IDs
Add error handling to real_size function.
Added test for Eyring 1st order linearly ramped temperature and updated test for Eyring 2nd order linearly ramped temperature in chempy.kinetics.tests.test_ode.py
Added test for sstable size in MB
Increased number of iterations for benchmarking from 1000 to 100000.
Fix for GLPK solver: changed conditional statement to check for non-zero integer or binary variables before calling glp_intopt.
Added doctest for normalize_arxiv_category function.
Added reversion signal for handling versioning.
Added _highest_parent_ attribute to Parameterized class
Added script file for validator manager
Fixed typo in hpc.py: changed 'pename' to 'name' in HPC_CONFIGS dictionary
Modified Potential.py to plot potential in X-Y instead of density.
Remove unnecessary 'except Exception' blocks in pkgrepo.py
Added new URL patterns for subscription and form validation views.
Fixed parsing tables in html.py
Added whitelist parameter to _process and _process_Stmt methods in SimEngineLightAIL class.
Bump version from 0.3.2 to 0.3.3 and change development status from Beta to Production/Stable.
Added import of sys module in inspector.py
Replace pypdf3 with pdfrw in upscale function.
Added URL and classifiers to setup.py
Bump version from 1.8.5 to 1.8.6.
Added log statement for resolving SRPM for RPM ID
Bump version number to 0.3.1.
Modified file.py to use 'file.append' instead of 'file.write' for appending lines to a file.
Update version numbers in _version.py
Replace MySQL query with SQLAlchemy query in search_unit_in_bibwords function.
HolodeckEnvironment: updated step() method to include a tick limit and added a custom exception message when the limit is reached. Also removed unnecessary code.
Improved documentation for approx() function in pytest/python.py
Modify create method in gce.py to use a different default value for tty parameter based on ssh_user
Added stop button functionality to AppManager.
Added is_local parameter to ApiConfig class and updated its logic.
Update country flag image URL to include country code in filename.
Fixed typo in Avatar template tag
Added function 'foo' to Motor class and updated autogen code.
Fix for detaching source in StripeObject
Adding a default value for the block_signing_key parameter in the Witness_update class.
PEP517: Add encoding conversion for build backend value in Pep517HookCaller._call_hook
Added __hash__ method to ContextOptions to implement caching
Fixed bug in TestFileCache where cache._getf(id) was not closing the file descriptor when expect_remove was True.
Added 'network' variable to etrago function and set it to original_network
Added configuration for denying current branch in git receive
Added check for existence of install config file before adding defaults
Renamed class ASN1_raw to ASN1_force and added a check for ASN1_Object in enc method.
Update to use self.init_app instead of init_app
Added charset-normalizer to install_requires in setup.py
Remove unnecessary 'required' field from KotlincPluginIdField.
Added support for Thermo precursor isolation window in MZMLIterator.
 Fixes #12345: Changes created_at column timezone to false for reports and location reports.
Fixed formatting issue in JSON generation for component parameters.
Update minion modules and states for remote compilation via ssh
Fix typo in couch_to_ntuple.py: changed 't.Fill()' to 't.Write()' and removed unnecessary 'file.Close()'.
ISSUE 88: Added set_session and created a new session for KerasRunner to resolve an issue with TensorFlow.
Added debug statements for duplicate waypoints and requested waypoints
Update ExactInvoice class to use ISO 8601 date format for 'From' and 'To' fields.
Added MixtureSameFamily to PYTREE_BLOCKLIST in jax_transformation_test.py
Added imports for re and requests, and updated schneier command to use requests instead of urllib2.
Added support for Google Bot, MSN Bot, Yahoo, Ask Jeeves, and AOL in UserAgentParser.
Fixed typo in VerboseOptimization initialization
Fix incorrect electrode ordering in EIT40 importer.
Replace ReceiptPDFDisplayView with ReceiptPDFView in test_views.py
Added logging and warning for missing name, email, and username fields in Enrich class.
Added special login handler for Arista devices
Changed float_prec to significant_figures in CifWriter init.
Update version number to 2.0.0
Update test_status.py to use cli_master instead of salt for pid checks on Windows.
Added function find_near_pol_angle to find nearest expected polar angle and rounded off polar angle to reduce number of unique models.
Modified tool environment to not require additional variables.
Added 'wafer.compare' to INSTALLED_APPS

Update introspection query to include optional fields for description and deprecated
Fixed logging service to correctly yield document contents
Update version number to 2.0.2 and add encoding parameter to open function in setup.py
Added check for file existence before using custom URL prefix in FormatedPhoto.url
Added __getitem__ method to Color class and updated Shape class initialization.
Fixed bug in AWSJobStore._sharedFileID: changed str(uuid.uuid5(self.sharedFileOwnerID, str(sharedFileName))) to str(uuid.uuid5(self.sharedFileOwnerID, sharedFileName))
Added timeout to get_next_data_elements_to_finish method to prevent hang during playback.
Fixing type error in int_to_roman function.
Update Context class to use a new Executor with a timeout parameter
Update setuptools requirement from >= 3.0.0 to >= 3.7.0
Extended format string parser to support 'i' specifier.
Update version number to 0.8.1 in setup.py
Update daemonize function to accept logdir argument and set it to /var/log/luigi by default.
Modified the code to use underscore notation for lattice vectors instead of attribute access.
Prohibit literalinclude directive for example DAGs and replace with exampleinclude directive.
Updated templatetags for admin interface to support Django 1.9+.
Removed unnecessary blank line in cflib/positioning/__init__.py
Added code to disable timeout for package installation and removal
Update BioGrid.py to set ingest source file version number
Removed clock difference tolerance and related methods.
Added print statements to BaseAttRelTestCase for better debugging.
Fixed formatting issue in SlurmCluster class.
Update subprocess imports
Modified PerformanceTestCase to test Cassandra's ability to handle large number of entities with different event histories.
 Fix JSONFormField to properly validate JSON input
Fixed typos in touch.py
Modified FunctionSolc class to link LoopExpression with Condition instead of StartLoop.
Update test_zinc_utils.py to correctly handle relative classpaths.
Added twine to dev extras in setup.py
Replace README with README.rst
Added retry mechanism for connecting to USB devices.
Added import statement for six.moves.collections_abc.Mapping
Bump version number to 0.0.6.
Modified zxbpp.py to update include file logic.
Bump version number to 0.14.1.
Removed unused imports and updated matplotlib usage.
Added check for '=' in environment variable assignments in SingularityParser.
Consumer: Changed auto_offset_reset from 'earliest' to 'fail'.
Relaxed check for unused5 field in PrimaryVolumeDescriptor to accommodate non-compliant ISOs.
Update plugin hub to use __dict__ instead of _active_plugins
Added support for Luciddream in buildbot.py
Update version number to 0.1.17
Update version number to 1.1.dev and change download base URL
Minor change: Moved touch_log_file(log_file) before log_file = os.path.join(edir, 'calc_%d.log' % job.id) for better performance.
Remove tearDown method from TestSetupCLI
Removed unnecessary try-except block for importing StringIO.
Added support for PyQt4 and PySide APIs in OffsetCalculatorMode.
Bump version from 0.0.6 to 0.0.7.
Fixed type hinting for create_status_callback function.
Fix issue with duplicate keys in probability map
Modified TurkeyExperiment class to simplify search string detection and improved readability.
Added boltons.funcutils import wraps and updated xarray import to version 0.15.0.
Update version of wcxf from 1.4.5 to 1.4.6 and update version of rundec from 0.5 to 0.5
Renamed 'costType' to 'cost_type' in ChampionSpellData and updated references in ChampionSpell class.
Modified help text for --absolute-time option in endpoint_is_activated function
Added property retrieval and printing for LocalClient.
Fix for getting ad item with station id and ad token
Update time measurement function for Python 2 and 3 compatibility.
AssertionError handling for negative sigma in NormalMessage constructor
Added import statements for matplotlib and cleared plot state in nx_qubit_layout function.
Update conf.py to use os.path.abspath for documentation root and add missing import statement for setup script.
Fixed type of DataFrame in TestMergeMulti test.
Update logging levels in server deployment mode
Fix for GPS heading calculation when speedE and speedN are negative.
Update vtk_to_meshio_type and meshio_to_vtk_type imports to use _vtk_common module.
Added import statement for matplotlib and updated show_hist function to use it.
Fixed calculation of scale and offset for CFScene.
Update copyright year and change DockerRegistry class name
Update version number to 0.0.0
Fixed bug in encode_jpeg function by swapping the dimensions of the input array before reshaping it.
Removed unnecessary Python3 future import and updated package name in setup.py
Fixed duplicate sample names in verifybamid.py
Added language field to PageIndex and updated prepare method to use it.
Removed unregister_actions() method from ExtendedController and moved it to child controllers.
Update invoice item tax rate calculation to exclude items with zero total
Added code to read image size from file instead of relying on Pillow library.
Added security warnings to settings.py to prevent running testapp in production.
Replace -C with -c in salt-cloud command
Implemented withdraw and deposit_address endpoints for Bittrex API.

Added opaque_keys.edx package to setup.py
Modified csrf_check to throw HTTP 403 error instead of HTTP 400 error for CSRF attacks.
Modified celerybeat schedule for intermittents commenter task
Update fweather.py to use more assertive language in error messages.
Added tests for BoundingBox.get_overlap_slices and updated test_bounding_box_slices.
Minor change in logging messages for download_per_well and download_feature_metadata methods.
utils.py: Fix for issue #12345 - Corrected error in FileLock.release() method.
Added check for no tasks found in Build.run_tests
Fixed issue with zipped file handling in AHIHSDFileHandler
Added list comprehension to unfollow non-followers.
Modified get_path method in view class to include an additional slash at the end of the path.
Added filter_params to CouchDBRiver init
Update version number to 0.2.1 in runner/precise_runner/__init__.py

Added error handling for ProgrammingError in Connection.get_cursor() method.
Fixed encoding for task experimental tags and dates.
Bumped version from 0.1.0a1 to 0.1.0a2.
Added functionality to generate OCSP keys and cache CRLs for Certificate Authority initialization.
Added ability to execute state machine until a specific state
Change sleep command from 1 to 0 in test_fork_exec
Modified SoilProfile class to use np.where instead of if-else statement for vertical effective stress calculation and deprecated old function.
Renamed 'entrypoint_waiter.Timeout' to 'EntrypointWaiterTimeout' and made it a subclass of Exception.
Added support for promoting positional arguments to keyword arguments in Callable class.
Allow for empty passwords in DatabaseInterface.
Added support for creating containers with CPU real-time options
Added test for YubiKey setup with ValidationService
Update Markov import and change 'Markov.Markov' to 'Markov'.
Added _result property to AsyncResult class and updated get() method to use it.
Modified BulkInserter with max_cache_size parameter in ClassicalHazardCalculator.
Added use_pk_only_optimization method to NestedHyperlinkedRelatedField and NestedHyperlinkedIdentityField
Rename 'Mac OS' to 'macOS' in OS_TO_CLASSIFIER_TAGS.
Added test for isinstance() with integer types.
Update impact file handling in RiabDock class.
Added parameter to exit_enable_mode method in SSHConnection class.
Added daemon option to Stream class and updated thread creation to respect daemon setting.
Update legend docstring to include custom handler information and note about increased linewidth
Update test_pydeps_colors to use -x enum with simpledeps('pdeps')
Added 'singlehtml' builder to copy asset files in add_assets function.
Added Python 3.6+ requirement check and exit script if not met.
Added __unicode__ method to KeyPublicationStatus model and corrected callback function in Report model.
Update version number to 0.3.7
Added error handling and fixups for ROOT logging inconsistencies.
Master and slave gateway startup ports updated from port number to host:port format.
Fix for invalid arguments handling in Commands class.
Fixed bug in get_current_user method of RequestHandler class.
Added support for multiple commits in autosynth
Removed unnecessary `asyncio.get_event_loop()` call in publish method.
Update GIT_ATTRIBUTES to use SHA256E instead of MD5E and add new largefile patterns.
Modified charting.py to adjust bar width and gap size for better visualization.
Added backports and tests to package data and updated README.rst, LICENSE, NEWS.txt, futurize.py, discover_tests.py, useful_links.txt, check_rst.sh.
Bump version number to 1.3.0
Added fallback for git describe --tags when running on Python 3.7 or newer
Remove deprecated duration attribute and methods.
Added verify=False to API initialization for asynchronous calls
Added test case for random control frame reception
Modified ImpactMergeDialog.py to use QUrl.fromLocalFile instead of QUrl for opening output directory.
Update course settings to use classrooms and student choice
Update pandoc.py to use 'utf-8' encoding when reading IPython notebooks.
Added test cases for TorConfig.create and TorConfig.contains
Rename _update_recurring_next_date() to _update_last_date_invoiced()
Added logic to handle empty results in RunSetResult.
Refactoring of handle_template function in mobanfile/templates.py. Changed _listing_directory_files_recusively to yield from _listing_directory_files_recursively and removed unnecessary variable.
Update prefix of router_yard from 'com' to 'master' in SaltEvent class.
Added test for missing grid mapping variable and updated parse_cf to log a warning when it cannot find the variable pointed to by grid_mapping.
Removed unnecessary line in rest_cherrypy/__init__.py
Update zypper.py to use raw string literals for version comparison
Added ContentItemRelation to ContentsEntryMixin
Modified arbiterdaemon.py to return a list of all states instead of just the last one.
Fixed issue with mouse position calculation
Modified suds/sudsobject.py: Removed unused method print_property and simplified print_object.
Added timed test case for SNMPTimeoutError in test_ThirdGetFailsAgain
Update install.py to use `runNpmInstall` with default values for development and npm flags.
Fix for incorrect result handling in vprof/runner.py
Update version number from 3.0 to 3.1.
Update particle positions and radii in optimize.py.
 Fixing assertion message for vaex.expresso.parse_expression
Added code to run the map editor and load the game state from a database file.
Added __unicode__ method to CssBackgroundAbstractBase and CssBackground classes
Added support for 2D WCS axes in wcs_utils.py.
Removed include_package_data=True in setup.py
Added 'upcoming' status to HealthEventFilter enum
Added support for version 13 in Application facade.
Added pyannote.metrics.segmentation.SegmentationPurityCoverageFMeasure class with tolerance and beta parameters.
Increase timeout for attach_stream_and_cancel test from 5 to 10 seconds
Added import statistics to test_titled_imports
Added OrderedDict to SphinxMapperBase for better performance.
ExternalTable: Use f-strings instead of format strings for table_name and s3
Update license URL in setup.py from 'https://github.com/Microsoft/LightGBM/blob/master/LICENSE' to 'https://github.com/Microsoft/LightGBM'
Update conftest.py to re-write CLI scripts every time tests are run.
Added support for Calamari and Radosgw logs in Ceph plugin.
Added retry mechanism for DataProc job retrieval in GCP Dataproc Hook.
Fix issue with find_symbol method in Exchange class
Removed unnecessary import statement for hsmm_subhmms.
Update test_io.py to use datetimes instead of year values in tdf_init and tdf_append indexes.
Added test for very large file support in ISO creation.
Update dependency version for snorkel-metal
Update dependencies to support Python 10 and later versions.
Added support for class and module fixtures in pytest_collection_modifyitems and pytest_runtest_protocol.
Added tests for num2words function in Portuguese (Brazilian) locale.
Update version number to 0.5.3 and add experimental iterative imputer from sklearn.
Added support for SSH known hosts with non-default ports
Added support for Fedora and RHEL in ipa_constants.py.
Added a new model chooser function to BayesianOptimization class.
Fix for setting minimum magnitude in readinput.py
Added test cases for negative temporal parameters in split function.
Update setup.py to use reStructuredText for long description
Update AppObj.py to set version for FlaskRestPlusAPIObject and register blueprint for web frontend
Update build.py to add --ignore flag for running tests
Removed unused variable DEFAULT_REACTIONS_DIR and updated parse_reactions method to use _context.resolve() instead of hardcoded directory path.
Added parameter 'default_image_config' to CompositeBase constructor and removed unnecessary code.
Removed unnecessary 'header_auth' parameter in Twitter API call
Added <div class='footer'> to waterfall.py
Modified login.py to support remote user authentication with environment variables.
Added optional `extract_dir` parameter to `ZipExtractor.from_path()` and `Extraction` class, allowing for extraction to a specific directory instead of a temporary one.
Update impact functions list and fix assertions in TestRegistry.py
Modified http.py to hide secret fields in POST data logs
Removed unnecessary import of 'dictionary' in favor of using 'f2p_dictionary' instead.
Added test for SQLite ENABLE_UPDATE_DELETE_LIMIT option.
Removed unnecessary ordering field in ArticleAdmin
Modified PreTrainedModel to use `os.path.isfile()` instead of `os.path.join()` to check for the existence of Flax weights file.
Fixed incorrect argument name in makeIdentifiers function call.
Added logging for task execution and deletion
Update blobstore tests to use new stubs
Update version number to 0.1.1 and add try-except block for importing setup tools.
Update version number to 1.0.9-dev
Changing memory format from GB to MiB and updating divisor accordingly.
Update setup.py to remove Python 2.7 specific dependencies.
Merge pull request #1234 from branch 'feature/new-ws-reverse-api' into branch 'main'.
Refactor deploy task to require product environment and add missing dependencies.
Added get_long_description function to read README.md file contents and updated setup() function to use it instead of codecs.open().
Added advanced XML manipulation methods using lxml library.
Added guard node check and modified circuit building command.
Added 'xliff' to FILE_TYPES in POEditorAPI
Fixed initialization of functionTagProcessor with default values for partially_qualified_names and include_function_signatures
Removed unnecessary permission for CancelAndReinstate action.
Added import statement for 'os' module in brian_full_network_test.py
Removed unused import statement and closed figure in plotRealImg function.
Updated BartModel to use shared input embeddings for encoder and decoder.
Modified workflow/coincidence.py to use inspiral_outs_dict instead of insp_files_dict for coincidence workflow.
Allow building from branch or tag in addition to commit SHA
Update license from Apache-2 to Apache-2.0
Update test_assertion.py to change expected output from '*- %d*' to '* 6*' for test TruncateExplanation.
Added check for absolute path in find_addresses() for non-Windows platforms
Update tests_require in setup.py to use Betamax 0.8 and add mock 1.0.1
Fixed login error handling for two factor authentication.
Added functionality to create and delete addresses in a wallet.
Fixed test_check_and_raise test case to correctly handle TimeoutError
Update CorrelogramViewModel to use `normalization` property instead of `_normalization` attribute, and update the view accordingly.
Restrict number of interpreter versions tested on MacOS for performance reasons.
Added direct fugacity calls for faster performance.
Added 'or {}' to ensure entities are created even if entities_json is empty
Remove unnecessary debug logs in Kubernetes check
Added support for parsing years and months in time strings.
Modified esptool.py to use float division instead of integer division for calculating flash block size.
Update _update_message and _get_auto_indent methods in PercentStyleMultiline class.
Added hidden attribute filtering to CRUDResponse.flatten() method.
Renamed component representation to include the function name for LambdaComponents
Bump version number to 0.21.0.
Added shutdown of Starmap in get_composite_source_model to save memory
Update colour scheme for lineage colours from Set1 to Dark2 and manually set None as grey.
FixedOffsetTimezone: simplified calculation of total_seconds
Added import statement for tempfile and updated remove_dir function to use tempfile.mkdtemp() instead of temp_util.temp_dir() for creating temporary directory on Windows.
Upgrade login account from version 1 to version 2.
Modified SAML2 metadata.py to support passing location and binding as a tuple instead of separate arguments.
Added retry logic for HTTPS requests and improved error handling in Discoverer class.
Added input validation for app name in Generate class of mantissacmd.py
Added check for int or PlayerID in lazy_player before registering player
Update environment.py to use config['postgresql'].get('superuser', {}) instead of hardcoded credentials for PostgreSQL connection.
Add chown command to set proper permissions for remote repository
Added check for empty cluster files and improved error handling in align_and_parse() function.
Added Linux to NATIVE_EOL_STYLE_MAP
Removed unnecessary 'self' argument from verify_certs method in rest_cherrypy.__init__.py
Update version number to 1.1.4
Update OMIM.py to correctly set omim_type for phenotype and predominantly phenotypes
Added guitar skip for NegativeBinomial distribution in DistributionsWorkWithAutoVectorizationTest.
Added wait for ready state complete when setting value of element on page loads.
Added support for 1D arrays in decision rules states.
Modified PingTransmitter class to use get_base_ping_command_list instead of hardcoded ping command
Set base prompt for IOSDriver
Update test file URL from {old_url} to {new_url}
Added support for taggit_selectize and updated dependencies.
Modified test_acid.py to use hszinc.parse() with single=True and removed unnecessary assertions.
Added field setting for positional arguments in Arguments class.
Added export LD_LIBRARY_PATH to pre_exec for intel composerxe 2015.2.164 compatibility.
Added logging for unsupported targets and skipped creation of MbedBoard instance for them.
Fix for push_and_tag_images to use == instead of in
Update XLM-R text preprocessing pipeline to use pre-trained sentencepiece model and vocabulary.
Update gerrit tests to include beaker prefix in change descriptions.
Modified Spectrum class to use inplace filtering by default.
Added import statements for time and unittest, and added a time.sleep(1) statement to AdsApiTestCase to avoid errors.
Update NER for Latin language
Added django-oauth-toolkit dependency and removed django-filter
Added save() method call to request object in test_homepage_requests_filled() method of FromHome class in threads/tests.py
Update version number to 0.1.3 and remove -b03 from download_url.
Added assertion for rundir in get_pidpath function.
Update lexers.py: Renamed 'm4' to 'M4' in MinimalisticM4Lexer class name and tokens dictionary.
JustinTV: Improved channel name extraction
Skip testing type of fields which are always strings
Added support for passing lists and tuples as column arguments to GroupBy.
Added ftp_socket_connected attribute and connected flag to FtpTarget class.
Added try-except block for importing fastavro.schema to handle missing package.
Added test for unreleased items going in unreleased release
Update version number to 0.8.1
Fix grammar in StructuredGridFilters.concatenate docstring
Fixed trac service login URL encoding.
Added close method to DaskResource class and added yield statement in dask_resource function.
Added Python 3.6 and 3.6 Notebooks to ServiceEnvironmentVersion and ServiceExecutionStatus enums.
Added error handling for missing 'stl' module
Added method as_dict to ConfigModel
Added psycopg2 to install_requires in setup.py.
Update timezone formatting in expand_template function.
Fixed issue with URL and body encoding in ExchangeNTLMAuthConnection.
Update SchedulerCeleryTasks from BUILDS_STOP to BUILD_JOBS_STOP in handle_new_build_job_status function.
Added buffering to sys.stdout when opening a file in write mode
Added fallback for Python 2.7 in text.py
Bump version number to 1.14.7
Added 'users' attribute to CommonNotificationMixin class
Added support for omitting multiple keys in Exchange.omit() method.
Added RegionRroposal and ProposalPostProcessor to the list of allowed classes in extract_scala_class.
RegtestDaemonService: updated make_request method to use 'generate' instead of 'setgenerate'
Update command-line interface for harvest
Update version number to 2.3.0.dev0
Reverted change in bct.py, removed return statement and added back randomizer_bin_und function.
Bump version number to 0.23.5.
Removed unittest import in test_aardvark.py
Added line break count argument to write_line_list and modified write_toc to use it.
Update pickleshare.py to use the correct version of the 'path' module.
Added HazardMapDBReader class to read hazard map data from the database and deserialize it into a data structure that can be passed to HazardMapXMLWriter.serialize to produce an XML file.
Update test cases to use bytes instead of strings for Unix FDs
Removed unnecessary debug log statement in urlencoded_processor function.
Simplify PandasQueryCompilerView by removing unnecessary reindexing and returning a new PandasQueryCompiler instance instead.
Added email.utils for formatting commit dates and added committer information to Git commits.
Added test cases for bugzilla queries.
Removed unnecessary print statement in load_catalog function.
Added data index handling in LinePlotDisplayLayersInspectorSection.
Added file argument to customized_warning function.
Split composite source model by calculation mode
Update import statement for unittest.mock and use mock.MagicMock instead of MagicMock.
Modified info command to require --datacenter option and added --ip-version option.
Replace ChromeDownloader with WgetDownloader in util.py
Added tests for DeVosSolarcell calculators returning units and special values.
Modified libextract/debug.py to update the decorator to print out the output value and add a prefix to the function name.
Added print statement in increaseamount_assemble_one_step and modified Rule creation in increaseamount_assemble_interactions_only.
Added import statement for 'sys' module in lektor_htmlmin.py
Bump version from 0.2.1 to 0.2.2
Fixed bug in MQTT disconnect handling by adding socket shutdown and close calls.
Added mark_safe to show_links method in EmailTemplateAdmin to allow for HTML tags in the email preview link.
Added support for passing positional and keyword arguments to handlers in IRC bot.
Update Meta class in BaseModelResource to set object_class to None
Added logging statement for loading Saltfile
Added support for adding forces as lists.
Removed import of 'get_view_name' and 'get_view_description' from rest_framework.utils.formatting, and instead imported 'get_view_name' and 'get_view_description' from rest_framework.views
Implemented Hessenberg decomposition via Householder transformations and fixed bug in scipy.linalg.hessenberg.
Added encoding for Python 3 compatibility.
Added test for deleting domain without removing others (#3475)
Added low-level lookup for function calls in convert_type_of_high_level_call
Added check for SSLWantReadError in HttpUA class
Set Windows taskbar icon
Added explicit encoding for Windows in pull function.
Skip foreign key columns in ModelConverter
Added logging statements for debugging purposes in decode_ReaderEventNotification and encode_ReaderEventNotification functions.
Added output file for pylint report
Fix for uploading file with correct file path and ID.
Modified database.py to cancel listener based on filter.
Updated admin.py to add created_at field to Statement model's list_display and list_filter.
Added error handling for missing abortConnection method in transport
Added new function 'new_random_token' for generating random tokens using Django's secret key and system random number generator.
Added cross parameter to spectrogram method
Added import statement for 'viz' module in suspect/_init_.py.
Upgrade BeautifulSoup version from 4.7.1 to 5.0.0.
Added error message details to D4S2Error raised in D4S2Api.
Added create_pending_tasks method and called it in Build class.
Added imports for random and hashlib in tests/minionswarm.py
Added Sentry404CatchMiddleware to catch ignorable 404 errors and prevent them from being sent to Sentry.
Added test for getting lldp information from a single interface using the _exec_lldp() function.
Removed unnecessary print statement in perturbations_higher_order.py
Added notification support for low battery alerts
Added private key retrieval and public key saving functionality to MasterKeys class.
Added get_items method to Cart class and added a list of items to the Cart instance.
Update regular expressions for parsing GitHub repository URLs and pull request numbers.
Update experimental.py to use list of str for excluded_entities and add rate_limit_timeout parameter
Added support for passing additional keyword arguments to get_statement_jsons_from_hashes.
Fixed incorrect import of models in fluent_comments.views.post_comment_ajax

Modified MonitoringCollector class to use signal.SIGKILL instead of signal.SIGINT for killing agents.
Added headers and Content-Type handling to AioHttpInterface's delete method.
Updated regular expressions for parsing notebook URL and directory
Added confidence intervals for healthcare per year and treated individuals in analysis.py.
Added test_tensorflow module
Update completer.py to improve triple string and comment handling.
Added future import and removed unnecessary function call.
Increased figure width from 0.8\textwidth to 0.95\textheight in run_data_clean_up.py.
Added import statement for 'sha3' from raiden.utils.
Added test cases for invalid arguments in test_simplesqlite.py
Fix for incorrect key reference in AESFuncs.minion_publish
Fixed division by 1024**3 instead of float(mem) // 1024**3
Fixed formatting issues in create_crl function in x509.py
Fix import statement for collections module.
Added import of builtins module and updated ExceptionGroup import to use dir(builtins) instead of __builtins__
Added test for resolving inexistent relationship and removed unused relationship
Fix for total power consumption reporting on legacy protocol.
Added request logging for Nova
Added import of helperservices in agol.__init__.py
Fix issue with deprecated_call and warns functions
Fix BinaryQuadraticModelStructureError message to correctly list all unsupported variables.
Added beautifulsoup4 as a dependency in setup.py
Added variable 'req' to WebApp class.
Added support for integer target labels in image classification evaluation.
Added support for Python 3.9 in LambdaMode
Added import of 'input' from 'django.utils.six.moves' in tenant_schemas/management/commands/__init__.py
Fixed bug in resource.py by correcting the assignment of local_session
Disable HTML output in QtConsole for DataFrame repr.
Update themed_tk.py to change the current working directory to the directory containing the theme files, and then source the pkgIndex.tcl file.
Added validation for API key in Transport class
Update __init__.py to use pkg_resources.resource_stream() for serving static files.
Update docgenerator.py to use ListSerializer and get_data_type for serializing fields.
Added clean_loaded_libraries and refresh_libraries to refresh_selected_state_machine
Update multiqc directory path in summary function of multiqc.py
Added LocalFileSystem import and updated checkout and transfer methods to use it.
Replace pybitcoin with virtualchain and update hash function
Removed unnecessary import statement for functools module in Windows.py.
Added method 'medians' to calculate medians of values with at least a certain number of values and minimal value.
Replace test_schmidt_normality with test_schmidt_seminormality in test_sphere_sph.py
Fixed bug in SimTypeFixedSizeArray._refine: now uses (self.elem_type.size/8) instead of self.elem_type.size
Modified run_image function in build.py to redirect stderr to STDOUT when running LXC commands.
Update GUI code to use PyQt4 and fix imports
Added force_color parameter to get_printer function in logger.py
Removed version check for Python 2.6 and older in setup_method and teardown_method of TestTestTest
Removed initialization from commander meta and added build, SystemLog, and pcb to HierarchicalShell.
Implemented softmax and logsumexp functions using pycuda.
Fixed typo in deci2sexa function
Replace json.load() with json.loads() in WDItemEngine class.
Added check for temperature == 0 in QuantyDockWidget, and updated nPsisAutoCheckBox, nPsisLineEdit, and calculation.temperature accordingly.
Added ssl parameter to interactive_server function
Fix for DisaggHazardCalculator to properly manage tasks
Fixed typo in HTMLRenderer.render method
Constellation: allow const_module to be a module or NoneType, and instruments to be a list-like or NoneType.
Modified PostgresStore class to use a smaller connection pool size.
Added test method test_03_clear_surroundings in TestToolboxClsGridLife.
Increased MAX_NAME_LENGTH from 50 to 47 in TravisSetupSteps.
Reduced wait time for restarting Spyder from 60 seconds to 20 seconds.
Update rauth version from 0.4.11 to 0.4.12.
Fix HTTP1Connection._parse_headers to strip leading \\r\\n from input data
Update WebDriver to use XPath instead of linkText and partialLinkText
Ensure that dictionary interface does not consume data of other types with an iterator interface
Update version number to 0.1.4 and remove unnecessary dependency on Pillow
Added source information to OTT initialization.
Added host.docker.internal host for Linux to emulate Mac and Windows behavior in Job model.
Update recipe.py to use input_processes instead of tasks for pipeline validation and execution.
win_service: module only works on Windows systems
Added custom theme options and sidebars to documentation configuration file.
Update regular expression for parsing oEmbed URLs to support non-ASCII characters.
Added InternalError class and updated hash function to handle unexpected error codes
Update Circuit.py to correctly handle blocking operations
Added matplotlib, wxPython, and PIL as install_requires for MAVProxy
Fix typo in test name: 'the_set_must_have_at_least_on_element' -> 'the_set_must_have_at_least_one_element'
Update REQUIRED package from 'serial' to 'pyserial' in setup.py
Update TIMEZONE_IDS to use repr() function for improved readability and consistency.
Refactor event classes and new_event function
Warn about deprecated console flow and suggest alternative local server flow
Update test_apiserver.py to use classmethod setUp and tearDown, and move api_server_process creation to a separate method.
Added checkpointing with disk saver and n_saved=2
Update get_user_data to use frozenset for storage argument
Added support for multilingual query sets by adding a new __class__ attribute to the QuerySet clone method.
Added test cases for Fossil integration
Added autoname function to Viaplay service
Added print statement in InterpolationTask to check the type of obs_interpolating_func and updated the LinearInterpolation class to use scipy.interpolate.interp1d with kind='linear' instead of numpy.interp for linear interpolation.
Added placeholders for mongo-profile and mongo-ss-txns in assess_deploy_webscale.py
Modified isChanged method in modify.py to correctly handle None values and different data types.
Added DocBookBodySubNode class.
Update test_cloudvolume.py to use .gz file extension in transfer test
Added support for sending emails using DatabaseEmailBackend. Modified send_messages method to use Email.objects.create() and added attachment support.
Added option to load map and show HUD in MAVProxy console.
Added support for command-level sleep and pane sleep in workspace builder
Added test for getting resource list from API
Added DEPRICATED_PACKAGE_NAME and updated EXT_MODULES to include ipopt_wrapper_ext.
Added x-axis and y-axis labels to the timeseries plot in followup.py.
Update MySQL connection URL for scicrunch registry syncing
Replace 'record[attr].fill_value' with 'getattr(record[attr], 'fill_value', None)' in RasterListSerializer.
Remove PrintSpace from page.get_Border() as GT now abides by the PAGE standard.
Remove unnecessary debug logs in HouseStatusMonitorEnable and HouseStatusMonitorDisable classes.
Removed unnecessary import 'urllib.parse' in define-profile-template.py
Fix for connecting to Docker server
Removed HipChat provider.
Added fallback implementation of filelock module for ICS NeoVi can backend
Replace . with \\\\1 in path for nested lookup
Removed DEBUG flag and updated subprocess call to use print statements instead of logging.
Added line self.canvas.focus_set() to HumanAgentHost class in human_action.py
Added support for Numpy array imports in JavaObjectUnmarshaller.
Sort SymbolIndex._PACKAGE_ALIASES before creating index
Added support for new webhook event types and updated signal handling.
Updated Protein class to use rstrip() method when translating DNA sequence.
Changed multiprocessing.Pool(processes=4) to multiprocessing.Pool(processes=multiprocessing.cpu_count()) in scan_dir function.
Added tooltip background color option.
Changed type of 'data' variable from list to string in pdf_show function to fix issues with transparent background.
Removed unnecessary rewriting of maven_metadata_path since the project is now hosted in Maven Central and uses a different convention for groupId/artifactId pairs.
Added punctuation exceptions to TextParser to permit certain punctuation characters within tokens.
Fix for argument error in ArgParser
Added functions for model and chart directories.
Fixed case sensitivity issue in Package.directory property by changing re.sub(r'[-_.]+', '-', self.name) to re.sub(r'[-_.]+', '-', self.name.lower())
Fixed test for find_and_modify with $inc and upsert=True
Update exception message for UnknownExtension
Added sorting to ChainedSelect widget
Update Service class to use --websocket-port instead of --port for CDP
Added functionality to OrderableManager to access next and previous ordered objects.
Removed unsupported Python 2.6 BZ2File implementation and updated to use the standard bz2 module instead.
Added mocking for mkdir function in test_blog.py
Added wrapping of return data in Caller.call() for consistent output
Fix bug in TestInfluxDB test: set result to None instead of assert None
Update setup.py to use py_modules instead of packages and remove unnecessary find_packages call.
Modified request schema fixture to use default strictness
Update dependencies: remove version pinning for moto and boto3, update google-compute-engine
Added rad2deg, deg2rad, minimum, and maximum functions to the vaex.dataset module.
Git: Set _git to bake with _tty_out=False by default
Added CRUD event creation and saving for m2m changed signal.

Added support for sending event webhooks with ECDSA signatures
Modified RandomizedSearchCV to use sampling with replacement for all parameters when at least one parameter is given as a distribution.
Fix grammar error handling in resolve_term_references
Added import statement for zipfile and updated export function to include zip file naming convention.
Added support for Unicode characters in format string checking.
Update API and Entity classes to use items() instead of iteritems()
Refactor accent handling in transform function.
Added upload settings for Red Hat Policy: upload_url='dropbox.redhat.com', upload_user='anonymous', upload_directory='/incoming'
Update win_dns_client.py to correctly handle server list reordering.
Update Task class to reset task_try_count to 0 on initialization
Add support for new GML namespace in EPSG code parser.
 Celery monitor: updated format_broker_info to as_uri in MonitorCommand class.
Fix error message in Stats plugin
Added initialize_placeholders function and updated step function to use it.
Update environment variables handling to preserve Unicode values on Windows.
Added --time as a required parameter for oqbugs.py and made some other changes to improve argument parsing.
Fix for Twisted's process support on Python 3
Removed unnecessary 'index' column in aggregate_losses and aggreg functions.
Added .gitignore file to S3Conf client.py and modified config file to include remote file.
Added light weight font option to text.py

Replace 'checkout' with 'reset --hard' in clone_git_repo function.
Added hz_actual variable to cpuinfo.py
Modify existing AR specs with new form values for selected analyses
HIDDevice: Changed hid.device() to create a new handle instead of opening an existing one.
Added fixture for testing argparse functionality and updated version number.
Replace FieldLenField('len', None, 'notdecoded', '@H', adjust=lambda pkt,x:x+8) with FieldLenField('len', None, 'notdecoded', '<H', adjust=lambda pkt,x:x+8)
Added synchronous query execution support via `execute_sync` function.
Added dependency on ujson and sh in setup.py
Added support for opening directories with default file manager on Linux.
Added shared memory parameter to StimelaJob class and updated job creation and start methods for podman and docker jobs.
Update version number to 0.5.0.
Added parsing function for samples in ZigBee class.
Added weight as an optional third argument to Infomap and updated print statement.
Modified OpenThreadTHCI class to set default domain prefix and updated configBorderRouter method.
Update URL patterns for Wafer pages
Fix for stack gathers in a list-of-traces
Added trailing slashes to local and remote paths in FileHandler.init() and updated routes dictionary key with _remote variable.
Added skipif condition for pytables test due to numpy version requirement.
Update import statements for ceph_deploy module
Added test cases for roundtrip escapechar and updated quoting paramterization in test_writer.py
Update MemoryView to display data as machine words by default (1 word per line) instead of 1 CPU word wide
Added check for too big PoEs in disaggregation calculator
SSHDetect: updated invalid responses list to include regular expressions
Fixed initialization of GoToLineDialog to use Qt.WindowTitleHint and Qt.WindowCloseButtonHint
Remove Python 2.7 support and add Future dependency for Python 3.5 and 3.6
Modify parameters of legislation
Added conversion of RGBA images to RGB for locateAll function
Update version number to 0.7.7rc1 in sandman/_init_.py
Update version number from 0.8.3 to 0.8.4
Added update_from_js parameter to create_array_binary_serialization function to handle updates from JavaScript side.
Added stop words in Kannada language
Update toolchain version to 4.8 for NDK r9.
Modified HTTPListener.py to correctly handle None values in deployment configuration.
Increased POSITIONS_PER_XML from 500 to 1000 in positionlabellerpart.py
Fix issue with cleandir() function.
Added option to disable reloader in Alerta app
Bump version from 42 to 43.
Fixed emitter for Active_QLabel clicked event.
Added test for deleting a category tree with children.
Update version number to 2.0.4
Added support for parsing rows with arbitrary column names and updated method naming convention.
Updated jar file name in sparkts test initialization.
Removed unnecessary map function in AddOnExecutor
Added version number printing and initialization support to webdriver test tools.
Modified read.py to use logger.warning instead of logger.warn for a specific message.

Update TEasyID3 test to handle multiple frames and undefined order of ID3.getall() return value.
Fix missing newline at end of file in interaction.py.
Update install_requires version of nionutils to 0.3.15.
Added InputBase class with constructor and properties for id, type, paramType, like, required, default, private, const, batchable, and maxbatchItem.
Added bytes conversion for delvals_regex in win_lgpo.py
Added progress bar for computing MD5 of large directories in collect_dir_cache method of RemoteLOCAL class.
Update libuv revision from 30c8be0 to 93877b1.
Update numpy version from 1.17.2 to 1.16.4
Added Apache 2.0 license to __init__.py
Fix incorrect count of entries in directories
Added tests for help subcommand and custom primary command
Added function 'bytes2NativeString' to util module.
Added environment option to ci_apextestsdb_upload command
Fix for plot_feature_importances function in sklearn module.
Update calculators initialization to use OQLITEIMPORT instead of OQLITEDEVPACKAGES.
Fixed repr formatting for ExhaustableQueue
Added new potentials and wrappers: CorotatingRotationWrapperPotential, GaussianAmplitudeWrapperPotential, and updated dependencies.
Bump version number to 0.13.5.
Bump version number to 0.9.0
Fixed assertion in UserModelTestCase for Python 3 compatibility.
Added minimum and maximum functions for kernel correction.
Fixed issue with StreamConn class by copying _handlers and _handler_symbols instead of modifying them directly.
Added import statements for openquake and fixed distributed computing code.
Added support for unstructured joins in BPMN workflows by raising a WorkflowException when two threads are waiting on the same sequence flow.
Update test_sendgrid.py to use Prisminstaller for installing Prism.
Allow user to specify event loop for run_app
Fixed debug logging for MakoTemplateAdapter's render method.
Update job listener to use JSON content type
Update URL and version number in setup.py
Added support for asynchronous execution of Salt commands using the 'yield' keyword.
Fix issue with _dimensions attribute in nc.py
Bump version from 0.7 to 0.9.
Modified _readline function in base.py to handle \r\n tokens correctly.
Refactor SocialAccount.get_provider_account() to use hasattr() instead of getattr() with try-except block.
Added long description and content type to setup.py
Skip test_ipv6_websocket_connection due to Travis errors when swapping between IPV 4 & 6
Added support for custom process identifier in MultiProcessValue
Renamed method 'find_in_parent' in class ParseResults from 'self' to 'sub'
Added randomization to test file readpartial in TestJFS.
Update print_figure_kwargs to use bbox_inches from environment variable SPY_BBOX_INCHES_O.
Update policy identifier and notice reference organization in PolicyInformation class.
Ensure label points are within image bounds
Added check for keep-alive in write_response method of HttpProtocol class
Update ClearFuncs to use get() method for retrieving ret_config from kwargs
Renamed gccarm to gcc_arm in build_project function.
Added typing to install_requires in setup.py
Update get_command_and_env function to use staging S3 bucket for CWL jobs
Enable Python subprocesses to work with expect functionality.
Added support for Hive datasets with custom separator character (0x01) in test_parse_many_cases.py
Rounded bar sizes and fixed color handling in Chart class.
Update Requests class to use http proxy instead of verify parameter
Added calculate_markov_matrix method to LexRank class.
Added 'bits' slot to StateVector class
Bump version from 1.2.0 to 1.3.0 and update dependencies
Added error handling for send_message_chat command in EjabberdctlBackend
Update version number to 0.13.0 and remove development label.
Added include_package_data=True to setup.py
Bump version from 1.0 to 1.1.dev0 and remove release date
Update CASClientWithSAMLV1 to inherit from CASClientV2 and add verify_ticket method with keyword arguments.
Fixed duplicate edges in graph construction.
Added register option to ReleaseMaker and set it to true by default
Fix for handling POST requests in WsgiServerTransport
Added support for parsing multipoint geometries with nested parentheses.
Added support for new Philips motion sensor models (SML003 and SML004) in custom device class PhilipsMotion.
Added unit tests for Canari project.
Added RemoveAdapterWithDevices method to Bluez5 mock object
Remove spurious dimensions from image data array.
Update version number to 0.2.5
Modified ClearlyServer class to use Celery instead of celery.app and updated constructor arguments.
Added support for random SP selection in spatial pooling.
Cache job store instance for unpickling
Updated README.rst with new version number and RtD links.
Added import statements for capnp and updated SpatialPooler class to use capnp schema.
Add xfail marker for cross-engine read/write NetCDF3 test due to issue #2050
Update EC2 instance types in test_sles_ec2_network.py
Removed version number from hosts.allow configuration
Fixed typo in _check_if_idle function in imap.py module.
Added support for asynchronous iteration by changing __aiter__ to an asynchronous function.
Update copyright year in conf.py to reflect current year (2017-2023).
Added deletion of files and directories
Added CI support for test_led.py
InfluxDBDriver: updated wait_for_line parameter in _exec method to use self.port instead of hardcoded 2
Added str() conversion for matplotlib version 1.1.1 in Trajectory.plot method.
Update version number to 0.2
Removed unnecessary print statements in ValidatorService.send_txn method.
Added method replace_secondary_ids to ReStructuredTextParser to replace alias IDs used in references with the corresponding primary ID.
Modified test_templatetags.py to correct data URI prefix.
Added support for decoding strings as UTF-8 when casting to Unicode
Renamed method 'hello' to 'hello' with self parameter in class ITestInterface.
Update format function in bfe_fulltext.py to remove unnecessary <small> tags and fix a syntax error.
Added __repr__ method to _TdsSession and _TdsSocket classes.
Modified Client class in connection.py to set TCP_NODELAY only when socket is not UDP or Unix and nodelay is True.
Added support for FAT file system in parted_partition module.
Update setup.py to use setuptools instead of distutils
Removed unused import: sharepa.helpers
Fixed bug in DocumentDetailFormView by properly saving form data to the document.
Utility function bit_length renamed to safe_string
Update cli.py to add -vv option for increased verbosity
Update forest.py to use sorted edges in graph_as_dot function.
Added support for parsing tracebacks and formatting notifications in BaseBackend.
ModulesEnv: updated 'which modulecmd' command to use decode('utf-8')
Removed unnecessary dependency on msvcr120.dll
Remove get and iget methods from Block and NonConsolidatableMixIn classes.
Added import of 'ImproperlyConfigured' exception from Django and updated raise statement to use it instead of 'Exception' in syntax_color.py.
Added terraform to tests_requirements in setup.py
Added support for Python 2.7 and updated version specs for Python 3.
Update setup.py to use set literals for excluding files with invalid encodings
Removed version setting for dataset file
Added on_leTitle_textEdited slot to update keywords list when title is changed
Added Pipfile.lock to list of lock files in is_lock_file function.
Added VERIFY_NULLFAIL flag to op_checksig function.
Refactor IceCreamDebugger to use passthrough instead of ret
Renamed `dewpt` parameter to `dewpoint` in function relative_humidity_from_dewpoint.
Added bounds for energy_lo, temp, and chem_potential parameters in uibei function.
Fixed getGeneHomolog to return the first hit instead of all hits, and removed unnecessary sorting.
Update icon settings for console and terminal
Added python_requires >= 3.4 to setup.py
Renamed function 'model_precision_recall_curve' to 'model_prerec_and_roc' and updated its docstring.
Fixed SSL certificate and key validation in webserver function.
Added exception handling for BadStatusLine in LimitedRequestQueueTests
Ensure SYN_DIR is available before opening URL
Update TreeTime class to use new inference method for ancestral sequences and prune short branches.
Reduced buffer length from 100 to 20 and changed object creation logic in test_object_spilling.py
Added support for friendly encoding in pageinfo.py
Update settings.py with new constants and updated Celery configuration.
Added retry mechanism for S3 Azure Blob bypass and fixed URL format for certificate errors.
Update version number to 0.1.2.
ASGIApp: updated stream_callback to handle StreamingHTTPResponse and removed unnecessary chunked header.
Replace self._results with self.results in ProblemSpec class.
Update Tika version from 1.19 to 1.22.
Added __del__ method to XidDevice class to properly close the connection and delete the object when it goes out of scope.
Fixed print statement for HTTPS and Unix domain.
Update test_fileserver.py to use support.unit instead of salttesting.
Update BBCiPlayer plugin to use appStoreState instead of episode
Added handling for legend_label and legend_field in ElementPlot.
Update version number to 1.3.12.
Added shard index check in DistributedSession constructor
Updated SAML_CONFIG settings in settings_ini.py to use config.get() instead of hardcoded file paths.
Fix for ConnectionDeadException
Updated feed processor to truncate feed title to 200 characters
Experimental support for GEM users: added ability to split calculation into tiles
Added utils import and updated predictor training to use SplitOutput.
Added support for finding packages with setuptools.find_packages() and updated setup.py to use it.
Added logging statement for virtual environment check and updated test to not fail when run without virtualenv.
Added missing register warning with address and size
Update test_tif2geojson.py to use 'in' and 'not in' instead of 'assertIn' and 'assertNotIn'.
Modified payloadBuilder method in MYQL class to accept optional format parameter and default to json if not provided.
Added check to skip purging if no packages need to be removed
Bump version from 0.9.24 to 0.9.25.
Added parameter 'func_name' to add_line_breakpoint function in Jinja2Debug.py
Update conformance test for PyGeoAPI to reflect new version of OGC API
Added proxy support to Exchange class by adding proxies parameter to request method and updated timeout calculation.
Bump version number to 0.7.1.
Reverse PEP and KO orders in Pairtrade algorithm.
Added count property to RounderBase class and updated _record_roundoff_error method to increment count. Also added average_roundoff property to return the average roundoff error.
Disallow whole-archive imports in MSVC linker.
Modified probematch.py to handle missing 'To' element in WS-Discovery messages
Upgrade psycopg2 dependency from 2.6.2 to >=2.3.1
Added import statement for errno module and updated chmod function to handle permissions correctly.
Added caching for _Props class to improve performance.
Update version number from 1.5.4 to 1.6.1.
Update documentation build to use sphinx.cmd.build.main() instead of sphinx.build_main()
Update jujupy/workloads.py: Add requests library and use it to get unit public IP. Also, update wait conditions for AllApplicationActive and AllApplicationWorkloads to use timeout parameter. (#12345)
Modified optimize parameters for MRD dimensionality reduction simulation.
Fixed inconsistent escaping in quote function.
Modified eighth_admin_dashboard_view to use select_related instead of prefetch_related for sponsors.
Added test cases for device get device properties and inheritance.
Added long description to README.md and updated version number in setup.py
Added type hints for function parameters in test_parsimonious.py
Fix type errors in test_dictregister.py
Improve performance of TokenList.remove() by using a list comprehension instead of LOOP(list.remove())
Rename and describe single supplementary zip file for Proof of Accessibility (PoA).
Modified tests/jenkins.py: removed unnecessary sleep statement and download_remote_reports block, updated cleaning and deleting VM logic.
Added log statement in Py3 class for command execution errors.
Fix for sending file with correct path separators in PyPI vendor dir.

Update download URL for PGPy
Added import statement for hashlib and added a new property get_atom_id that generates an atom id based on the export's slug, max_visible_items, and photo format using hashlib. Also updated the url method to use the new property get_atom_id.
Update minimum_needs_postprocessor.py to use decimal value for evacuation percentage instead of integer.
Update version number to 0.8.15.

Added assertions for job_id in JobTestCase
Remove unconfigured settings from FlaskLoader
Bump version number from 3.7.0 to 3.8.0.dev
Replace instruction argument string with hash
Update base URL of OpenStreetMap API from http to https
Update version number to 1.4.11 and update copyright year.
Modified sleep function in Bittrex class to use call_rate variable instead of hardcoded value.
Fixed issue with preorder expiration check in BlockstackDB
Added cumulative returns since policy update for PPO trainer.
Added support for freezing columns with all cells evaluated.
Replace _embed_code_links with a version that uses full function names
Added support for Osram Flex RGBW LED strip and outdoor version.
Update version number to 0.2.1b2 and add support for Python 2.6 and 3.0 in setup.py
Added support for 3 Hz channels in FIL_reader.read_all() method.
Added support for numeric format strings in SheetFreqTable.
Added get_autosync_payee method to LedgerPython and HLedger classes.
Update waveforms.py to draw line strip for spike waveforms only when n_spikes is not None and greater than 0.
Update schema_external.py to use dict constructor for share configuration
Removed environment variable OQ_ENGINE_MODE
Rename test method from test_raises_when_zeep_web_fault to test_raises_when_zeep_exception
Update state.py to use _salt_.kwargs instead of **kwargs
Added SupportedServices.register_resource_view(queryset.model, resource_view) to register resource views for querysets with a specified model.
Fixed download_lectures function to use 'with' statement instead of 'open' and 'close' for writing m3u file.
Modified EClass.py to replace metaclass with a new implementation.
Added support for getting system unit string from base units.
Update binary_heap.py to use 'smaller than' instead of 'smaller its children'.
Modified experiment.py to use os.path.join for constructing file paths instead of concatenating strings.
VCR: Convert hosts_to_ignore from list to set and update ignore_localhost handling.
Removed unnecessary comments and cleaned up code
Exclude tests and tasks from package list
Update _APITOOLS_VERSION from 0.5.9 to 0.5.10.
Added TestReactorMixin to EndpointMixin
Fix for negative scenario ID
Added gradient support for scipy.linalg.sqrtm and updated solve_triangular gradients.
Update data decode tests to use utf-8 encoding and handle non-string, non-data-structure items correctly.
Remove unnecessary slave_okay property and clarify documentation for max_pool_size
Renamed 'new' method to 'create' and updated its parameters to include an optional 'attach' parameter for streaming stdout/stderr. Also removed unnecessary 'type' parameter and updated return type to match.
Fix http://crbug.com/394454 on cros by disabling testIFrame on windows.
Update version number to 0.6.0.dev20191212
Fixed bug in BoundingBox.get_overlap_slices: raised ValueError for negative ixmin or iymin
Added test for output format inclusion
Upgraded py26compat.py to support Python 2.7+
Bump version number to 0.3.2.
Added code to update important extra selections when setting color scheme in EditorStack.
Update EMS API client initialization to use a custom transport class and enable caching.
Added deepcopy import and modified createTree function to use deepcopy for caching.
Update training script to use MultiStepLR instead of StepLR for learning rate scheduling and added resume functionality.
Update clirundal.py to use instrument configuration from CLI instead of images
Upgrade controller and enable OS upgrade
Added support for custom Fabric environments through ~/.fabsetup-custom directory (#80644)
Added aspect ratio calculation to Beam class.
Added PIL warning removal to cleanup function.
Removed unused function mock_raw_event and simplified alternate function.
Added labels and help texts to AssignShiftForm
Removed unnecessary code in LlandControl.py
Added check for calc_id in datastore before querying database
Update AMIProtocol to correctly detect protocol version.
WorkerPool: Use `num_workers` instead of `len(workers)` for getting the number of workers.
Remove unused method purge_db and update no_such_entity to use do_404 instead of raise NotImplementedError
Added 'pylab' to MOCK_MODULES in doc/conf.py
Added import statement for fgivenx and added skipIf clause for test_param_logx_diagram and test_bs_param_dists to skip the tests when fgivenx is not available.
Added support for Kuijken & Gilmore (1989) potential in KGPotential class.
Changed InfoFmtString from '...%(message)s' to '-> %(message)s' in kappa/context.py
Added textupdate widget to AsynOutportPart's create_tags method.
Added hash computation for Pipfile.lock
Added connection_lost method to StunProtocol class and removed unnecessary code.
Fixed return type of _precache_zipimporters to match actual behavior
Update on_teams_members_removed method to include team info parameter.
Added error handling for starting worker in beam boot
Update KaplanMeierFitter class to use exponential Greenwood formula for confidence intervals instead of the previous method referenced in a PDF document.
Increased maximum number of results for geohash lookups from 10 to 20.
Added support for infinite expressions by importing math and using it to check for infiniteness in addition and multiplication operations.
Added try-except block to handle errors when creating directory paths in verify_env function.
Added newline character to 'task' field in NIDMResults constructor.
Modify _toggle_term_protect function to use get_node instead of LocalClient.
Added child watcher for asyncio event loop
Update brain.py to improve response selection logic.
Update cloudvolume/paths.py to use cloudfiles.paths.to_https_protocol instead of hardcoded replaces for gs, s3, and matrix protocols.
Upgrade upgrade_master.py to use configurable configuration file name.
Added check for zero target packages and updated pytest arguments
Added URL for Django-ICAL project
Update pagerank_weighted.py to use xrange instead of range for iteration.
Normalize model output and separate output values for Sobol analysis.
Fixed typo in release_resources method name.
Added 🕹️ and 🎞️ to double_width_emojis list in rich_helper.py
Fixed game entity parsing
Update setup.py to use distutils.core for setup
Added support for libxslt in the FreeBSD deployment script.
Removed unnecessary named capture groups in split_address and split_housenumber functions.
Added support for remote images in RinohBuilder.
Replace dj_database_url.parse() with url = dj_database_url.parse()
Replace backslashes in taxon names with underscores to avoid confusion with file paths.
Added 'has_role' method to PersonEntity, and updated 'role_in' method to use it.
Update to use getfullargspec in Python 2 and 3
Set _datastore to None if not provided in constructor
