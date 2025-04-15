Detached a buffer from the chunk adapter and disconnected the port from the NodeMap for device {0}.
Replace get_list with list_buckets in S3Manager.
Added check for is_ec2_instance before reporting error
Added profiling for server loop using cProfile
Renamed second unit to MetricUnit and added 'sec' and 'seconds' to its aliases.
Added support for Python 3.x in Coconut parser.
Fix bug in testStore: Simplify assertion in checkRemove
Update format string in FileTest to use {0} instead of {1} for consistency.
Removed unnecessary print statements in unescape function.
Added code to print traceback when importing modules fails in MAVProxy.
Update version number to 0.2.0 and copyright year to 2021
Update pandas repository path and benchmark timespan.
Added option to disable cross-seeding and set additional explicit label for cross-seeding.
MAGICC7 and above: logging level controlled with debug argument.
Added exception handling for renormalization of spectrum in Interpreter class.
Removed unnecessary close() call in get_file_chunk_consumer() function.
Added _MIN_LEN variable and updated register_bpdu_type method to include _MIN_LEN check
Added reporting for digital inputs
Fix for incorrect encoding in SortedHostCollection.pop() method.
Remove legislative input parameter from Column constructor
Added logging and exception handling for Braintree vault information retrieval in payments_billing view.
Update version format in OBO parser
Added color values and reference link to powerline-bash.py
Modified Address class to use classmethod for b58decode and fixed parameter error handling.
Update doc_tree.py to handle missing YAML blocks
Removed print statement and updated assertions in ToolboxHtmlToolsTest.
Update send_file function to use cache_timeout and last_modified from quart.static instead of hardcoded values.
Modified heartbeater function to log debug instead of exception
Fixed typo in docstring of Method class
Added probe service and constants
Added group parameter to list_metafeatures method and added support for landmarking metafeatures.
Updated regular expression for recognizing CASA files
Added check for _connection before refreshing schema and node list
Fix for mkdir_p to avoid creating duplicate directories
Added release to sphinx and wheel dependencies.
Update version number to 1.0.2
Added darwin to list of supported platforms in platforms() function.
Added code to ensure client disconnection and reconnection in interacting_function_client1

Bump version from 0.10.0 to 0.10.1.
Added test for index dot slices
Added SoS_Exporter to kernel.py
Added data files for scripts and updated setup.py
Added flake8 directive to disable linting warnings in Django models.py file.
Added MACOSX_DEPLOYMENT_TARGET=10.7 environment variable to setup.py to enable building wheels for OSX 10.7 and later.
Allow window_shell to be optional in Session constructor
Upgrade wheel from 0.29 to 0.30
Update matplotlib.use('agg') to matplotlib.use('Agg') in syncpr_tests.py
Added a comment to the parse function in ttlfmt.py to indicate that we don't need to reset saved parsed streams to the point where they fail.
Added option to specify AWS profile and instance type
Added captureoutput tag for capturing template output into a variable
Disable close_on_teardown for PyMemcache in test_flask_pymemcache.py
Update version number to 1.4.0
Added blur option to heatmap with time test
Added test for code blocks and fixed syntax boundaries in CMark parser.
Added support for two new voices: Joanna (en-US) and Mizuki (ja-JP).
Added detection for Read the Docs environment and adjusted sys.path accordingly.
Added support for Dask-based NinjoTIFF writer.
Update GitFormula to fetch and merge instead of clone and checkout
Increased thumbnail width and height in image.py
Added logging for unexpected engine output in XBoardProtocol
Added bool_only parameter to generic.py, allowing for selection of only boolean columns.
Update Action.to_rest to include dp as an argument
Fix for k8s tests failing due to missing images
Added test for deleting domain without removing others (#3475)
Added skipping of read-only attribute fields when resource is non-existent
Added root_url parameter to JSHost constructor in host.py
Update open function in Path class to use keyword arguments and improve error handling.
Update Renderer to use partials instead of loader
Update conftest.py to properly handle homely module loading.
Replace OpenSSL.SSL.Connection with Connection
Handle property definitions in _impl phase
Added support for ellipses in ShapeList.
Added test for empty response content in head requests
Fix Varscan VCF output by replacing sample names with those from input BAMs
Added support for passing configuration to message processor constructor.
Simplify boolean AND expressions by flattening them and checking if we are left with a single argument.
Added import of Board in Session class to ease circular import issues and fixed type hints for Board and DebugProbeServer.
Expand exception types in _TemporalAttribute.parse_datetime to include TypeError.
Update generate_json_docs.py to use 'gcloud' instead of 'gcloud-python' in clean_source_path function.
Removed segmentize_on_clip parameter in InputData constructor.
Added tests for check_request with allowed caps and describeprocess
Modified BackboneWithFPN class to use a different constructor for ResNet backbones with FPN.
Added exception for chopping datetime type in Calendar class.
Added test for reading empty arrays with various dtypes using pyarrow
Added check for list of permission repositories and updated logic for getting permission from the first repository in the list.
Added get_token_features function and updated gram_label and gender_match_label functions to use it.
ArrayEditor: Allow selecting and copying whole columns or rows.
Added skip marker for test_fuse_projections
Added new properties to StageDescription class: CacheDataEncrypted, CacheTtlInSeconds, CachingEnabled, DataTraceEnabled, LoggingLevel, MetricsEnabled, ThrottlingBurstLimit, and ThrottlingRateLimit. Removed DeploymentId property.
Removed unnecessary prints in SqlaTable.order_by
Added shortcuts for calling methods on Lens objects
Removed source_parsers and updated source_suffix to only include .rst files.
Update wait_for_droplet_to_be_created to use droplet UIDs instead of IDs
Add error handling to real_size function.
Added test for Eyring 1st order linearly ramped temperature and updated test for Eyring 2nd order linearly ramped temperature in chempy.kinetics.tests.test_ode.py
Added test for sstable size in MB
Increased number of iterations in benchmarks.py from 1000 to 100000.
Fix for GLPK solver: changed condition for calling glp_intopt() to handle cases where there are no integer or binary variables.
Added doctest for normalize_arxiv_category function.
Added reversion.signals import and connected handle_version function to post_revision_commit signal.
Added _highest_parent_ attribute to Parameterized class and updated its usage in the code.
Added script file for validator manager
Fixed typo in hpc.py: changed 'pename' to 'name' in HPC_CONFIGS dictionary
Modified Potential.py to plot potential in X-Y instead of density.
Remove unnecessary try-except block in pkgrepo.py and add a check for the existence of a package repository.
Added new URL patterns for classic form subscription and form data validation.
Fixed parsing tables in html.py
Added whitelist parameter to _process and _process_Stmt methods in SimEngineLightAIL class.
Bump version from 0.3.2 to 0.3.3 and change development status from Beta to Production/Stable.
Added import of sys module in inspector.py
Update upscale method from pypdf3 to pdfrw
Added URL and classifiers to setup.py
Bump version from 1.8.5 to 1.8.6.
Added log statement for resolving SRPM for RPM ID
Bump version number to 0.3.1.
Fixed bug in file.py: removed unnecessary 'len()' check in append() function.
Update version numbers in _version.py
Replace MySQL query with SQLAlchemy query in search_unit_in_bibwords function.
Update HolodeckEnvironment class to use a more informative exception message when the tick limit is reached.
Update docstrings for approx() function to clarify its behavior and comparison with other floating-point comparison methods.
Modified create method in gce.py to use a different default value for tty parameter based on ssh_user.
Added stop button functionality to AppManager.
Added is_local parameter to ApiConfig class and updated its logic.
Updated country flag image URL replacement pattern to support two-letter country codes.
Fixed typo in Avatar template tag
Added function 'foo' to class Motor
Fix for detaching source in StripeObject
Adding a default value for the block_signing_key parameter in the Witness_update class.
PEP 517: Add encoding conversion for build backend value in Pep517HookCaller._call_hook
Added __hash__ method to ContextOptions to implement caching
Update test_cache.py to correctly handle file cache removal.
Added a new line 'network = original_network' in the 'etrago' function to keep track of the original network before making changes.
Update configuration for denyCurrentBranch in git
Added check for existence of install config file before adding defaults
Renamed class ASN1_raw to ASN1_force and added a check for ASN1_Object in enc method.
Update to use self.init_app instead of init_app
Added charset-normalizer to install_requires in setup.py.
Remove unnecessary 'required' field from KotlincPluginIdField.
Added support for Thermo instrument format in ProteomicsParsers.py.
Fixed timezone for created_at column in Report and LocationReport models.
Fixed syntax error in JSON generation for component parameters.
Update minion modules and states for remote compilation via ssh
Replace Fill() with Write() in couch_to_ntuple.py
ISSUE 88: Added set_session and self.sess for KerasRunner
Added print statements for duplicate waypoints and requested waypoints
Fixed issue with invoice date formatting.
Added MixtureSameFamily to PYTREE_BLOCKLIST in jax_transformation_test.py
Added imports for re and requests, and updated schneier command to use requests instead of urllib2, and also added html_parser import
Added support for Google Bot, MSN Bot, Yahoo, Ask Jeeves, and AOL browser in UserAgentParser.
Removed unnecessary assignment of ipython_notebook to self.verbose in VerboseOptimization constructor.
Fix for wrong ordering in _average_swapped_current_injections
Renamed view from ReceiptPDFDisplayView to ReceiptPDFView in test_views.py
Added logging for missing name, email, and username fields in Enrich class.
Added special login handler for Arista devices to handle 'Last login: ' message.
Updated CifWriter init method to use 'significant_figures' instead of 'float_prec' for formatting floats, and updated docstring accordingly.
Update version number to 2.0.0
Update test_status.py to use cli_master instead of salt for pid checks on Windows.
Added function find_near_pol_angle to find nearest expected polar angle and rounded off polar angle to reduce number of unique models.
Modified tool environment method to include additional information on recommended usage.
Added 'wafer.compare' to INSTALLED_APPS
Added support for multiple seeds in prep_bootstrap.
Update introspection query to include optional fields for description and deprecated
Fixed logging service to correctly yield document in hits
Update version number to 2.0.2 and add encoding parameter to open function in setup.py
Fixed photo URL prefix for non-existent images.
Added __getitem__ method to Color class and fixed Shape class initialization
Fixed bug in AWSJobStore._sharedFileID: changed str(uuid.uuid5(self.sharedFileOwnerID, str(sharedFileName))) to str(uuid.uuid5(self.sharedFileOwnerID, sharedFileName))
Added timeout to get_next_data_elements_to_finish method to prevent hang during playback.
Fixed type hinting in find_minmax and int_to_roman functions.
Update Context class to use a new Executor with a timeout parameter
Update __requires__ to setuptools >= 3.7.0
Extended format string parser to support 'i' specifier.
Update version number to 0.8.1 in setup.py
Allow daemonize() to accept a custom log directory
Fixed a bug in the IStructure class where the lattice parameters were not being correctly passed from the primitive structure to the derived structure.
Fixed typo in documentation for 'exampleinclude' directive
Update admin interface templatetags to use Django 1.9+ syntax
Removed unnecessary line in positioning/_init_.py
Added code to disable timeout for package installation and removal
Update BioGrid.py to set ingest source file version number
Removed clock difference tolerance and related methods.
Added print statements to BaseAttRelTestCase to improve error reporting.
Fixed typo in SlurmCluster class
Removed unnecessary import statement for 'call' from subprocess module.
Update performance test to use range(1, 5) and change num_beats calculation.
Fixed JSONFormField to properly validate JSON input
Fixed typo in touch.py
Modified FunctionSolc class to link LoopExpression with Condition instead of StartLoop.
Update test_zinc_utils.py to correctly handle relative classpaths.
Added twine to dev extras in setup.py
Replace README with README.rst
Added retry mechanism for connecting to USB devices.
Added import statement for six.moves.collections_abc.Mapping
Update version number to 0.0.6.
Added support for including files in ZXBPP programs.
Update version number to 0.14.1
Removed unnecessary imports and updated matplotlib usage.
Added check for '=' in environment variable assignments in SingularityParser.

Relaxed check for unused5 field in PrimaryVolumeDescriptor to accommodate non-compliant ISOs.
Update plugin hub to use __dict__ instead of _active_plugins
Added support for Luciddream in buildbot.py
Update version number to 0.1.17
Update version number to 1.1.dev and remove release candidate label.
Moved touch_log_file call before log_file assignment.
Removed unnecessary tearDown method in TestSetupCLI
Removed import of cStringIO and updated import of StringIO in srtfile.py
Added support for PyQt4 and PySide APIs in OffsetCalculatorMode.
Bump version from 0.0.6 to 0.0.7.
Update game_interface.py to use CFUNCTYPE(None, c_int, c_int) instead of CFUNCTYPE(ctypes.c_int, ctypes.c_int)
Fix issue with duplicate keys in probability map
Modified TurkeyExperiment class to simplify search string detection and improved readability.
Added boltons.funcutils import wraps and updated xarray import to version 0.16.2.
Update version of wcxf from 1.4.5 to 1.4.6 and update version of rundec from 0.5 to 0.5
Renamed 'costType' to 'cost_type' in ChampionSpellData and updated references in ChampionSpell class.
Modified help text for --absolute-time option in endpoint_is_activated function.
Added property retrieval and printing for LocalClient in test.py.
Fixed get_ad_item method to include stationId in ad metadata
Utility module updates for Python 2 and 3 compatibility.
Added assertion error for negative sigma in NormalMessage constructor.
Added import statements for matplotlib and cleared plot state in nx_qubit_layout function.
Update conf.py to use os.path.abspath for documentation root and add missing import statement for sys.
Update DataFrame creation in TestMergeMulti test case
Minor logging changes in server_deployment_mode and main functions.
Added calculation of lon_scale and updated formula for gps_heading in process_flightgear function.
Update vtk modules and meshio version
Added import statement for matplotlib and updated show_hist function to use it.
Fixed calculation of scale and offset for CFScene.
Update copyright year and change DockerRegistry class name
Update version number to 0.0.0
Modified cloudvolume/chunks.py to change the reshape operation for encoding JPEG images.
Removed unnecessary Python3 future import and updated package name in setup.py
Fixed duplicate sample names in verifybamid.py
Added language field to PageIndex and updated prepare method to use it.
ExtendedController: Removed unregister_actions() method and updated __action_registered_controllers removal.
Update invoice item tax rate calculation to exclude items with zero total
Added code to read image dimensions from file and set width and height attributes for Image model.
Added security warnings to settings.py to prevent running testapp in production.
Update ShellCase.py to use -c instead of -C for salt-cloud command.
Implemented withdraw and deposit_address endpoints for Bittrex API.
Added last_stdout and last_stderr variables to Connection class for storing stdout and stderr of exec_command.
Added opaque_keys.edx package to setup.py
Modified csrf_check to throw HTTP 403 error instead of HTTP 400 error for CSRF attacks.
Modified Celerybeat schedule for intermittents-commenter task
Update fweather.py to use more assertive language in error messages.
Added tests for BoundingBox.get_overlap_slices and updated test_bounding_box_slices.
Modified logging messages in download_per_well and download_feature_metadata methods to use f-strings instead of string concatenation.
utils.py: Fix for issue #12345 - Corrected error in FileLock.release() method.
Added task detection for build jobs and raised error if no tasks are found
Fixed issue with zipped file handling in AHIHSDFileHandler
Added list comprehension to unfollow non-followers.
Modified get_path method in view class to include '/' at the end of the path
Added filter_params to CouchDBRiver init
Update version number to 0.2.1 in runner/precise_runner/__init__.py
Bump version from 2.1.2 to 2.1.3
Added handling for ProgrammingError in Connection.get_cursor() method.
Update utils.py to correctly encode task dates and tags.
Update version number from 0.1.0a1 to 0.1.0a2.
Added functionality to generate OCSP keys and cache CRLs for Certificate Authority initialization.
Added ability to execute state machine until a specific state
Change sleep time from 1 second to 0 seconds in test_fork_exec.
Added new method 'vert_eff_stress' for determining vertical effective stress in SoilProfile class, and deprecated old method 'vertical_effective_stress'. Also updated 'determine_vertical_effective_stress' method to use np.where instead of if-else statement.
Renamed 'entrypoint_waiter.Timeout' to 'EntrypointWaiterTimeout' and made it a subclass of Exception.
Added support for promoting positional arguments to keyword arguments in Callable class.
Allow for empty passwords in DatabaseInterface
Added support for CPU real-time options in container creation
Added test for YubiKey setup with ValidationService
Replace Markov.Markov with Markov and fix minor syntax errors.
Added _result property to AsyncResult class and updated get() method to use it.
Increased max cache size for BulkInserter in ClassicalHazardCalculator from _CURVE_CACHE_SIZE to max_cache_size=_CURVE_CACHE_SIZE
Added use_pk_only_optimization method to NestedHyperlinkedRelatedField and NestedHyperlinkedIdentityField to disable primary key optimization.
Update OS name from 'mac_os' to 'macOS' in V2_TO_V1_MAP.
Added test for isinstance() with integer types.
Modified RiabDock class to use impactLayer instead of impactFile for getting file name and metadata.
Added parameter to exit_enable_mode method in SSHConnection class.
Added daemon option to Stream class and updated thread creation to respect daemon setting.
Update legend docstring to include custom handler information and note about increased linewidth.
Update test_pydeps_colors to use -x enum with simpledeps('pdeps')
Added 'singlehtml' builder to copy asset files in add_assets function.
Added Python 3.6+ requirement check and exit script if not met.
Added __unicode__ method to KeyPublicationStatus model and corrected callback function in Report model.
Bump version from 0.3.6 to 0.3.7.
Added error handling and fixups for ROOT logging inconsistencies.
Gateway: updated log messages to include hostname and port
Fix for invalid arguments handling in Commands class.
Fixed bug in get_current_user method of RequestHandler class.
Added support for multiple commits in autosynth
Fixed typo in publish method
Update GIT_ATTRIBUTES to use SHA256E instead of MD5E and add new largefile patterns.
Increased bar gap from 10% to 5% in Chart class.
Added backports and tests to package data and updated README.rst, LICENSE, NEWS.txt, futurize.py, discover_tests.py, useful_links.txt, and check_rst.sh.
Update version number to (1, 3, 0)
Added fallback for git describe --tags when running on Python 3.8 or later
Remove deprecated duration attribute and methods.
Added verify=False to API initialization for asynchronous calls
Added test case for random control frame reception
Fixed issue with opening output directory in ImpactMergeDialog by changing QUrl construction.
Update course settings page to use get_id() instead of directly accessing the course ID, and fix a bug where the use_classrooms setting was not being updated correctly.
Added encoding='utf-8' to pandoc.py file opening.
Added test for creating a Tor configuration object with a fake protocol implementation.
Rename _update_recurring_next_date() to _update_last_date_invoiced()
Added check for empty results and added at least one status column
Replace list comprehension with generator expression in handle_template function.
Update prefix of router_yard from 'com' to 'master' in SaltEvent class.
Added test for missing grid mapping variable and updated parse_cf to log a warning when it cannot find the variable pointed to by grid_mapping.
Remove unnecessary Content-Type header in REST API
Fixed regular expression pattern to use raw string literal for better readability.
Added ContentItemRelation to ContentsEntryMixin
Fixed get_all_states method in ArbiterDaemon to return a list of states instead of a single state.
Fixed issue with mouse position calculation when hovering over window borders.
Modified suds/sudsobject.py: Removed unused method print_property and simplified print_object.
Added timed and raises annotations for test_ThirdGetFailsAgain.
Update install.py to use `runNpmInstall` with default values for `dev` and `npm` when `opts` is None.
Fix for incorrect result handling in vprof/runner.py
Update version number from 3.0 to 3.1.
Update particle positions and radii in optimize.py.
Fixed bug in expresso.py by adding a more informative error message when encountering non-expression input.
Added code to run the map editor and load the game state from a database file.
Added __unicode__ method to CssBackgroundAbstractBase and CssBackground classes.
Added support for 2-dimensional WCS coordinates.
Removed include_package_data=True from setup.py
Added 'upcoming' status to HealthEventFilter enum
Added support for version 13 in Application facade.
Added pyannote.metrics.segmentation.SegmentationPurityCoverageFMeasure class with tolerance and beta parameters.
Increased timeout for attach_stream_and_cancel test from 5 seconds to 10 seconds.
Added import statistics to test_titled_imports
Added support for ordered dictionaries in SphinxMapperBase
ExternalTable: Use f-strings instead of format strings for table_name and s3
Update license URL in setup.py from 'https://github.com/Microsoft/LightGBM/blob/master/LICENSE' to 'https://github.com/Microsoft/LightGBM'
Update conftest.py to re-write CLI scripts every time tests are run.
Added support for Calamari and Radosgw logs in Ceph plugin.
Added retry mechanism for DataProc job retrieval
Fix issue with find_symbol method in Exchange class
Removed unnecessary import statement for hsmm_subhmms.
Update test_io.py to append data to timeseries dataframe and update metadata.
Added test for very large file support in ISO creation.
Upgrade snorkel-metal dependency to >=0.1.7
Update dependencies to support Python 10 and later versions.
Allow fixtures with scope 'class' or 'module' to be used in test functions.
Added tests for num2words function in Portuguese (Brazilian) locale.
Update version number to 0.5.3 and add experimental iterative imputer from sklearn.
Modified ssh.py to allow for specifying a non-default port when adding a known host
Added support for Fedora and RHEL in ipa_constants.py.
Added a new model chooser function to BayesianOptimization class.
Fix for setting minimum magnitude in readinput.py
Added test cases for negative temporal parameters in split function.
Update setup.py to use reStructuredText for long description
Update AppObj.py to set version for Flask RestPlus API object and register blueprint for web frontend.
Update build.py to add --ignore flag for running tests
Remove unused variable and directory, and simplify reaction parsing.
Added parameter 'default_image_config' to CompositeBase constructor and removed unnecessary code.
Removed unnecessary access_token_secret parameter in Twitter API call
Added newline character in WaterfallStatusResource class
Allow remote user authentication with HTTP headers and environment variables.
Added optional `extract_dir` parameter to `ZipExtractor.from_path()` and `Extraction` class, allowing for extraction to a specific directory instead of a temporary one.
Update impact functions list and fix assertions in TestRegistry.py
Modified http.py to hide secret fields in POST data logs
Added import statement for 'f2p_dictionary' in __init__.py
Added test for SQLite ENABLE_UPDATE_DELETE_LIMIT option.
Removed unnecessary ordering field in ArticleAdmin
Added check for Flax weights file existence before loading TensorFlow weights.
Fixed typo in makeIdentifiers function call: changed outputDir to outputDir
Added logging for task execution and deleted tasks, and updated exception handling.
Removed unnecessary imports and updated blobstore stub registration.
Update version number to 0.1.1 and add try-except block for importing setuptools or distutils
Update version number to 1.0.9-dev
Changed unit of measurement for available memory from GB to MiB.
Removed Python 2.7 specific dependencies
Merge pull request #1234 from branch 'feature/new-ws-reverse-api' into 'main'.\n\nThis commit includes changes to the CQHttp class in aiocqhttp//__init__.py. The changes include removing the handle_ws_reverse_event and handle_ws_reverse_api functions, and instead using a single handle_ws_reverse function for both event and api websockets. Additionally, the connected_ws_reverse_api_clients dictionary has been removed. These changes are part of the new ws reverse api feature.
Refactor deploy task to require product environment and add missing dependencies.
Update README.md and switch to using get_long_description() instead of codecs.open() for reading it.
Added advanced XML manipulation methods using lxml library.
EXTENDCIRCUIT command modified to use a guard node for the first hop when building a circuit.
Added support for xliff files in POEditorAPI.
Fixed initialization of functionTagProcessor with default values for partially_qualified_names and include_function_signatures
Removed unnecessary permission for CancelAndReinstate action.
Added import statement for 'os' module in brian_full_network_test.py
Removed unused import statement and closed figure in plotRealImg function.
Added shared embeddings for encoder and decoder in BartModel.
Modified workflow/coincidence.py to use inspiral_outs_dict instead of insp_files_dict for storing output files.
Allow building from branch or tag in addition to commit SHA
Update license from Apache-2 to Apache-2.0
Update test_assertion.py to change expected output from '*- %d*' to '* 6*' for test TruncateExplanation.
Added check for absolute path in find_addresses() for non-Windows platforms.
Update tests_require in setup.py to use Betamax 0.8 and add mock 1.0.1
Added check for two factor authentication during login
Added function CreateAddress to create a new address in the UserWallet class, and updated DeleteAddress function to use the new CreateAddress function.
Update test_elapsedtimer.py to remove unnecessary 'break' statement and add missing 'assert' statement.
CorrelogramViewModel: updated imported_params and normalization setter.
Restrict Python interpreter versions tested on MacOS for performance reasons.
Added direct fugacity calls for faster performance.
Added fallback for missing entity JSON in simulation
Removed unnecessary log statement in Kubernetes check
Added support for parsing years and months in time strings.
Fixed calculation of number of blocks for flashing.
Update _update_message and _get_auto_indent methods in PercentStyleMultiline class.
Added hidden attribute filtering to CRUDResponse.flatten() method.
Renamed component representation to include the function name for LambdaComponents
Update version number to 0.21.0
Added shutdown of parallel.Starmap to save memory in get_composite_source_model
Update colour scheme for lineage colours from Set1 to Dark2 and manually set None as grey.
FixedOffsetTimezone: simplified calculation of total_seconds by removing unnecessary conversion to days and using a single division operation.
Added import statement for tempfile and updated remove_dir function to use tempfile.mkdtemp() instead of temp_util.temp_dir() for creating a temporary directory on Windows.
Upgrade login account from version 1 to version 2.
Modified SAML2 metadata.py to support passing location and binding as a tuple instead of separate arguments.
Added retry logic for HTTPS requests and improved error handling in Discoverer class.
Added input validation for app name in Generate class
Added check for int or PlayerID in lazy_player before registering player
Update environment.py to use config['postgresql'].get('superuser', {}) instead of hardcoded credentials for PostgreSQL connection.
Added chown command to set proper permissions for remote repository
Added check for empty cluster files and improved error handling in align_and_parse() function.
Added Linux to NATIVE_EOL_STYLE_MAP
Removed unnecessary 'self' argument from verify_certs function in rest_cherrypy.__init__.py
Update version number to 1.1.4
Update OMIM.py to correctly set omim_type for phenotype and predominantly phenotypes
Added guitar skip for NegativeBinomial distribution in DistributionsWorkWithAutoVectorizationTest.
Added wait for ready state complete when setting value of an element
Added support for 1D arrays in decision rules states.
Modified PingTransmitter class to use get_base_ping_command_list instead of hardcoded ping command
Set base prompt for IOSDriver
Update test file URL from https://github.com/danielhrisca/asammdf/files/1593570/test.demo.zip to https://github.com/danielhrisca/asammdf/files/1594267/test.demo.zip
Update taggit dependencies
Fix parse issue in test_acid.py by using hszinc.parse with single=True and removing unnecessary assertions.
Added field setting for positional arguments in Arguments class.
Added export LD_LIBRARY_PATH to pre_exec for intel composer xe 2015.2.164
Added logging and support for unsupported targets in MbedBoard class.
Fix push_and_tag_images to use == instead of in
Update XLM-R text preprocessing pipeline to use pre-trained sentencepiece model and vocabulary.
Update gerrit tests to include beaker prefix in change descriptions.
Modified Spectrum class to use inplace filtering by default.
Added import statements for time and unittest, and added a time.sleep(1) statement to AdsApiTestCase to avoid errors.
Update NER for Latin language
Added dependency on django-oauth-toolkit and removed django-filter
Added save method call to Request object in test_homepage_requests_filled test case.
Update version number to 0.1.3 and remove -b03 from download_url.
Added assertion for rundir in get_pidpath function.
Update lexers.py: Renamed 'm4' to 'M4' in MinimalisticM4Lexer class name and tokens dictionary.
JustinTV: Improved channel name extraction
Skip testing type of fields which are always strings in _parse function.
Added support for passing lists and tuples as column arguments to GroupBy.
Added ftp_socket_connected attribute and connected method to FtpTarget class.
Added try-except block for importing fastavro.schema to handle missing package.
Added test for unreleased items going in unreleased release
Update version number to 0.8.1
Concatenate a structured grid to this grid.
Added support for authenticated Trac API requests by updating the URI construction.
Added close method to DaskResource class and added yield statement in dask_resource function.
Added Python 3.6 and 3.6 notebooks to ServiceEnvironmentVersion and ServiceExecutionStatus enums.
Added error handling for missing 'stl' module
Added method as_dict to ConfigModel
Added psycopg2 to install_requires in setup.py.
Update timezone formatting in expand_template function.
Fixed issue with URL and body encoding in ExchangeNTLMAuthConnection.
Modify SchedulerCeleryTasks from BUILDS_STOP to BUILD_JOBS_STOP in handle_new_build_job_status function.
Added buffering to sys.stdout when opening a file in write mode.
Added backport for Python 2.7 in text.py
Bump version number to 1.14.7
Added users field to CommonNotificationMixin and updated render method to include users in the notification data.
Fixed type hinting for Exchange.omit and Exchange.unique
Added RegionProposal and ProposalPostProcessor to the list of allowed classes in extract_scala_class.
Modified RegtestDaemonService to use 'generate' instead of 'setgenerate' for sending raw transactions.
Update command-line interface for harvest
Update version number to 2.3.0.dev0
Revised bct.py: added return statement in randomizer_bin_und function.
Bump version number to 0.23.5.
Removed unittest import in test_aardvark.py
Modified ReadmeMaker class to use line_break_count parameter in write_line_list method and added line_break_count parameter to write_toc method.
Update import statement for path module
Added HazardMapDBReader class to read hazard map data from the database and deserialize it into a data structure that can be passed to HazardMapXMLWriter.serialize to produce an XML file.
Update test cases to use bytes instead of strings for Unix FDs
Removed unnecessary logger statement in urlencoded_processor function.
Simplify PandasQueryCompilerView by removing unnecessary reindexing and returning a new PandasQueryCompiler instance instead.
Added email.utils.formatdate and git.Actor imports, and updated commit logic to use renku version {0} and set author date and committer information.
Added test cases for bugzilla queries.
Removed unnecessary print statement in load_catalog function.
Added data index handling in LinePlotDisplayLayersInspectorSection.
Added parameter 'file' and 'line' to customized_warning function.
Modified readinput.py to change the split_all condition for ucerf calculation modes.
Added import of unittest.mock and updated patching of jupytext.contentsmanager.load_jupytext_config to use mock.MagicMock instead of MagicMock.
Added required flag to --datacenter option in info command
Replace ChromeDownloader with WgetDownloader in util.py
Added tests for DeVosSolarcell calculators returning units and special values.
Update debug.py to print out output value and add a prefix to the function name.
Added print statement in increaseamount_assemble_one_step and modified Rule creation in increaseamount_assemble_interactions_only.
Added import statement for 'sys' module in lektor_htmlmin.py
Bump version from 0.2.1 to 0.2.2
Fixed bug in MQTT disconnect handling by adding socket shutdown and close calls.
Added mark_safe to show_links method in EmailTemplateAdmin to allow for HTML links
Added support for passing positional and keyword arguments to handlers in IRC bot.
Update Meta class in BaseModelResource to set object_class to None
Added logging statement for loading Saltfile
Added support for adding forces as lists, and updated __add__ and __radd__ methods to handle lists.
Removed import of 'get_view_name' and 'get_view_description' from rest_framework.utils.formatting, and instead imported 'get_view_name' and 'get_view_description' from rest_framework.views
Added hessenberg function for Householder transformations and fixed bug in scipy.linalg.hessenberg
Added encoding for Python 3 compatibility.
Fixes #3475: Add test for deleting a domain without removing others
Added low-level lookup for function calls in convert_type_of_high_level_call.
Added check for SSLWantReadError in HttpUA class
Added code to set Windows taskbar icon for Conda Manager.
Added explicit encoding for Windows in pull function.
Skip foreign key columns when there are no overridden columns
Update ReaderEventNotification encoding
Added output file for pylint report
Fix for uploading file with correct file path and file ID
Modified database.py to cancel listener based on filter.
Updated StatementAdmin to include created_at in list_display and list_filter.
Added error handling for missing abortConnection method in transport
Added new function 'new_random_token' for generating random tokens using Django's secret key and system random number generator.
Added cross-spectrum calculation option to TimeSeries.spectrogram method.
Added import statement for 'viz' module in suspect/_init_.py.
Update BeautifulSoup import to use bs4 instead of lxml.cssselect
Added error message detail to D4S2Error raised in D4S2Api.
Added create_pending_tasks method and called it in Build class.
Added imports for random and hashlib in tests/minionswarm.py
Added check for ignorable 404s in Sentry404CatchMiddleware
Add test for getting lldp information from a single interface using exec_command
Removed unnecessary print statement in perturbations_higher_order.py
Fixed notification bug in BatteryChecker class.
Added private key retrieval and saving functionality to MasterKeys class.
Added get_items method to Cart class and added a list of items to the Cart instance.
Update regular expressions for parsing GitHub repository URLs and pull request numbers.
Update experimental.py to use list of str for excluded_entities and add rate_limit_timeout parameter
Added support for passing additional keyword arguments to get_statement_jsons_from_hashes.
Fix for getting model by name instead of hardcoded app name. Changed import from django.db.models to django.apps.registry.apps.get_model for better reliability and maintainability.
Renamed image variable from 'image' to 'image_name' in InsideBuilder class.
Modified MonitoringCollector class to use signal.SIGKILL instead of signal.SIGINT for killing agents.
Added headers and Content-Type handling to AioHttpInterface's delete method.
Updated regular expressions for parsing notebook URL and directory
Format healthcare access and treated individuals print statements in analysis.py.
Added test module for TensorFlow.
Update completer.py to improve triple string and comment handling.
Added future import and removed unnecessary 'exists' import in entity_test.py.
Increased width of float caption in run_data_clean_up.py from 0.8\textwidth to 0.95\textheight.
Added import statement for 'sha3' from raiden.utils in test_restapi.py.
Update test_simplesqlite.py to include additional test cases for invalid arguments
Fixed typo in AESFuncs.minion_publish
Fixed division by 1024**3 instead of float(mem) // 1024**3
Remove unnecessary str() cast in create_crl() function.
Update import statement for collections module to use abc instead of collections.
Added import of builtins module and updated ExceptionGroup import to use dir(builtins) instead of __builtins__.
Added test for resolving inexistent relationships
Fixed issue with TotalConsumption on legacy protocol, now returns 'N/A' instead of failing and creating log mess.
Added requests logging to Nova cloud provider
Added import of helperservices in agol.__init__.py
Fix issue with incorrect type hinting in warns function.
Fix BinaryQuadraticModelStructureError message to correctly list all unsupported variables.
Added beautifulsoup4 as a dependency in setup.py
Added variable 'req' to WebApp class.
Added support for int target labels in image classification evaluation.
Update runtime enum in LambdaMode to include python3.9
Added import of 'input' module in tenant_schemas/management/commands/__init__.py
Replace db._local.instance with db.session in Resource class
Disable HTML output in QtConsole for DataFrame repr.
Added code to change the current working directory to the directory containing the theme files, and then source the pkgIndex.tcl file.
Added validation for API key in Transport class
Update __init__.py to use pkg_resources.resource_stream() for serving static files.
Update docgenerator.py to use ListSerializer and get_data_type for serializer names.
Added cleaning of loaded libraries and refresh of state machines in response to cancelation of selected state machine refresh.
Update multiqc directory path in summary function of multiqc.py
Added LocalFileSystem import and modified transfer method to handle relative paths in StageCache.
Update virtualchain dependencies and use keylib for private key generation.
Removed unnecessary import statement for functools module in Windows.py.
Added method 'medians' to calculate medians of values with at least a certain number of values and minimal value, and updated 'variances' method to log debug message.
Replace test_schmidt_normality with test_schmidt_seminormality in test_sphere_sph.py
Fixed bug in SimTypeFixedSizeArray._refine: now uses (self.elem_type.size/8) instead of self.elem_type.size for computing address increment.
Modified run_image function in build.py to redirect stderr to STDOUT when running LXC commands.
Update GUI code to use PyQt4 and fix imports.
Added 'force_color' parameter to get_printer function in logger.py.
Removed version check for Python 2.6 and older in setup_method and teardown_method of TestTestTest class.
Removed unused import statement and updated shell commands.
Added logsumexp and softmax functions to pycuda_ops.softmax.py, and updated cross_entropy function to use logsumexp.
Fix for deci2sexa function: removed unnecessary 'None' comparisons and fixed a bug where upper limit was not being respected.
Update to use json.loads() instead of json.load()
Added temperature-based logic for nPsisAuto checkbox and line edit updates in QuantyDockWidget.
Added ssl parameter to interactive_server function
Fix for DisaggHazardCalculator to properly manage tasks
Update HTMLRenderer to use render_inner() without rstrip()
Update Constellation class to accept a module or NoneType for const_module and a list-like or NoneType for instruments.
Update PostgresStore.connect to use a smaller connection pool size
Added test method test_03_clear_surroundings in TestToolboxClsGridLife.
Increased MAX_NAME_LENGTH from 50 to 47 in TravisSetupSteps.
Reduced wait time for resetting subprocess from 60 seconds to 20 seconds in restart_app.py.
Update version number to 0.4.12
Update HTTP1Connection._parse_headers to strip leading whitespace from headers
Update WebDriver to use XPath instead of linkText and partialLinkText
Added check to ensure that interface does not consume data of other types with an iterator interface in DictInterface class.
Bump version number to 0.1.4 and remove unnecessary dependency on Pillow
Added source information to OTT initialization.
Added host.docker.internal host for Linux to emulate Mac and Windows behavior in Job model.
Update recipe.py to use input_processes instead of tasks for pipeline validation and execution.
win_service: module only works on Windows systems
Added custom theme options and sidebars to documentation configuration file.
Update regular expression for parsing oEmbed URLs
Added InternalError class and updated hash function to throw an exception on error.
Fix grammar and clarity in Circuit.py comment
Added dependencies on matplotlib, wxPython, and PIL for MAVProxy.
Fix test for DisaggregationBinaryMatrixXMLWriterTestCase
Update REQUIRED package from 'serial' to 'pyserial' in setup.py
Update timezones list to use repr() function for better readability.
Refactor event handling to use descriptors and introduce PutEvent and DeleteEvent classes.
Warn user about deprecated console flow and suggest alternative local server flow
Update test_apiserver.py to use classmethod setUpClass and tearDownClass for starting and terminating the API server process, and move the clear_users method to the TestApiServer class.
Added checkpointing with disk saver and n_saved=2
Modified get_user_data function to use frozenset for storage parameter
Added support for multilingual query sets by adding a new _clone() method that takes a `klass` parameter and updates the cloned object's class to match.
Added test cases for Fossil integration
Added autoname function to Viaplay service
Added print statement in InterpolationTask to check the type of obs_interpolating_func and updated LinearInterpolation class to use _library instead of library for scipy.interpolate.interp1d
Added placeholders for MongoDB profile and SS/TXNS settings in assess_deploy_webscale.py
Modified VisiData modify.py file to correctly handle None values in isChanged method.
Added DocBookBodySubNode class.
Added file compression to transfer test.
Added support for sending emails using DatabaseEmailBackend. Modified send_messages method to use Email.objects.create instead of creating a new MIMEMessage object. Also added a return statement at the end of the method to return the number of emails sent.
Added 'Show Map' and 'Show HUD' menu items to ConsoleModule.
Added support for command-level sleep and pane sleep in workspace builder
Add test for getting resource list from API
Added DEPRICATED_PACKAGE_NAME and updated package list in setup.py.
Added x-axis and y-axis labels to the timeseries plot in followup.py.
Update MySQL connection URL for scicrunch registry syncing
Replace 'record[attr].fill_value' with 'getattr(record[attr], 'fill_value', None)' in RasterListSerializer.
Remove PrintSpace from page.get_Border() as GT now abides by the PAGE standard.
Remove unnecessary debug logs in HouseStatusMonitorEnable and HouseStatusMonitorDisable classes.
Removed unnecessary import 'urllib.parse' in define-profile-template.py
Fix connectivity issue with Docker server
Removed HipChat provider.
Added fallback implementation of filelock module for ICS NeoVi can backend
Replace . with \\\\1 in path for nested lookup
PDF: removed DEBUG flag and changed log statements to print statements.
Added line self.canvas.focus_set() to HumanAgentHost class in human_action.py
Added support for Numpy array imports in JavaObjectUnmarshaller.
Sort SymbolIndex._PACKAGE_ALIASES before creating indexes
Update webhooks.py to use set instead of dict for WEBHOOK_EVENT_TYPES and add missing signal for vault.credit-card.updated
Update protein sequence translation method to remove trailing asterisks.
Update multiprocessing.Pool(processes=4) to multiprocessing.Pool(processes=multiprocessing.cpu_count()) in scan_dir function.
Added TOOLTIP_BACKGROUND_COLOR constant and updated tooltip background color to #ffffe0.
PDF_display.py: Changed pixel area data type from 'bytearray' to 'str' for compatibility with new function requirement.
Removed maven_metadata_path from files_to_rewrite since it is now hosted in Maven Central and incompatible with the convention used by Maven Central.
Allow certain punctuation characters within tokens
Fix for argument error in ArgParser.parse_args() method.
Added functions models_dir and charts_dir to misc.py
Fixed case sensitivity issue in Package.directory property by changing re.sub(r'[-_.]+', '-', self.name) to re.sub(r'[-_.]+', '-', self.name.lower())
Fixed test for find_and_modify with $inc and upsert=True
Update exception message for UnknownExtension
Added sort method to ChainedSelect widget
Update selenium/webdriver/firefox/service.py: Remove unnecessary --websocket-port argument and use --connect-existing instead.
Added additional functionality to OrderableManager to access next and previous ordered objects.
Remove unused code for Python 2.6 BZ2File support.
Added mocking for mkdir function in test_blog.py
Added wrapping of return data in Caller.call() for consistent output
Fix typo in TestInfluxDB.thd_postStatsValue assertion.
Update setup.py to use py_modules instead of packages and fix dependencies.
Update request schema fixture to use default strictness
Update dependencies: remove version pinning for moto and boto3, update google-compute-engine
Added rad2deg, deg2rad, minimum, and maximum functions to vaex.dataset.py
Git: Set _git to bake with _tty_out=False by default.
Added CRUD event creation for m2m changed signal.
Fixed comparison of compositions with different number of elements
Added support for sending event webhooks with ECDSA signatures.
Allow sampling with replacement for all parameters when at least one parameter is given as a distribution.
Fix grammar error handling in resolve_term_references
Added import statement for zipfile and updated export function to include zip file naming convention.
Added support for Unicode characters in format string checking.
Update API and Entity classes to use items() instead of iteritems()
Refactor accent handling in transform function.
Added upload settings for Red Hat Policy: upload_url='dropbox.redhat.com', upload_user='anonymous', upload_directory='/incoming'
Update win_dns_client.py to correctly handle server list reordering.
Fixed task try count increment in Task class
Add support for additional EPSG code types using a dictionary-based approach.
Replace format_broker_info with as_uri in MonitorCommand
Update stats display to show 'No stats found' instead of erroring
Added initialize_placeholders function and updated step function to use it.
Fixed issue with environment variables not being properly set in CommandLineJob on Windows.
Modified argparse actions for oqbugs.py
Remove unnecessary skip tests for processExit on Python 3 and non-POSIX platforms.
Removed unnecessary 'index' column in aggregate_losses and aggreg functions.
Added .gitignore file to S3CONF directory and updated config file to include remote file path.
Added light weight font option to text.py
Bump version from 2.2.0 to 2.2.1
Modified util.py to use 'git reset --hard' instead of 'git checkout' for checking out branches.
Added hz_actual variable to cpuinfo.py
Modify existing AR specs with new form values for selected analyses
Update HIDDevice.open() method to use hid.device() instead of hid.device(None, None, path=self.path)
Added fixture for testing argparse functionality and updated version number.
Update RadioTap.len field to use '<H' instead of '@H' for adjusting field length.
Added synchronous query execution support by introducing execute_sync function.
Added dependencies on ujson and sh in setup.py.
Added support for opening directories with default file manager on Linux.
Added shared memory parameter to StimelaJob class and updated job creation and start methods for Podman and Docker jobs.
Update version number to 0.5.0.
Added parsing function for samples in ZigBee class.
Added weight as an optional third argument to Infomap and updated print statement.
Modified OpenThreadTHCI class to set default domain prefix and updated configBorderRouter method.
Update wafer/pages/urls.py: Fix redirect for index page
Fix stack gathering in test_groups_gather_equivalence
Added trailing slashes to local and remote paths in FileHandler.init() and updated routes dictionary with _remote instead of remote.
Added skipif condition for pytables conda build package with numpy 1.15 requirement
Update import statements for ceph_deploy module
Added test cases for roundtrip escapechar and updated quoting parameter in UnicodeWriter.
Update MemoryView to display data as machine words by default (1 word per line) instead of 1 CPU word wide.
Added check for too big PoEs in disaggregation calculator and updated bin edges for sites with multiple RLZs.
SSHDetect: updated invalid responses list to include regular expressions.
Fixed initialization of GoToLineDialog to use Qt.WindowTitleHint and Qt.WindowCloseButtonHint
Update setup.py to support Python 2.7 and remove unnecessary dependencies.
Modify parameters of legislation
Added conversion of RGBA images to RGB for locateAll function
Update version number to 0.7.7rc1.
Update version number from 0.8.3 to 0.8.4
Added update_from_js parameter to create_array_binary_serialization function to handle updates from JavaScript side.
Added stop words in Kannada language
Update toolchain version to 4.8 for NDK r9.
Fixed handling of non-None values in make_configuration_file
Increased POSITIONS_PER_XML from 500 to 1000 in positionlabellerpart.py
Fix issue with cleandir() function.
Added option to disable reloader in Alerta app
Bump version from 42 to 43.
Fixed emitter for Active_QLabel clicked event.
Added test for deleting a category tree with children.
Update version number to 2.0.4
Added support for parsing rows with multiple values per cell, and added methods for getting unique values in a column.
Update spark timeseries jar file path
Removed unnecessary map function in AddOnExecutor
Added version number printing and initialization support to webdriver test tools.
Updated logging level for warning messages in read.py
Remove unused variable 'has_ref' in Instance class.
Fix for issue #12345: Ensure that EasyID3.getall returns keys in a consistent order.
Fix for computing MNL interaction dataset.
Update install_requires version of nionutils to 0.3.15.
Added InputBase class with constructor and properties for id, type, paramType, like, required, default, private, const, batchable, and maxbatchItem.
Added bytes conversion for delvals_regex in win_lgpo.py
Fixed issue with large directory handling in RemoteLOCAL.py
Update libuv revision from 30c8be0 to 93877b1.
Update numpy version from 1.17.2 to 1.16.4.
Added Apache 2.0 license to __init__.py
Fixes behavior of directory count to include downloads directory.
Added tests for help subcommand and custom primary command
Added function 'bytes2NativeString' to util module.
Added environment option to ci_apextestsdb_upload command.
Fix for plot_feature_importances function in sklearn module.
Update calculators initialization to use OQLITEIMPORT instead of OQLITEDEVPACKAGES.
Update repr format for ExhaustableQueue and Worker classes
Added CorotatingRotationWrapperPotential and GaussianAmplitudeWrapperPotential to galpy.potential_src, and updated imports accordingly.
Update version number to 0.13.5.
Bump version number to 0.9.0
Fixed assertion in UserModelTestCase for Python 3 compatibility.
Implemented symmetric kernel computation for MNN Graph.
Fixed bug in StreamConn class by copying _handlers and _handler_symbols instead of modifying them directly.
Added import statements for openquake and updated utils_tasks.distribute to use celery for post-processing computation.
Added support for unstructured joins in BPMN workflows by raising a WorkflowException when two threads are waiting on the same sequence flow.
Update Prism installation script to use Prisminstaller
Added option to pass custom event loop to run_app() method
Fixed debug logging for MakoTemplateAdapter's render method.
Replace 'Content-Type' with string literal 'application/json' in job_listener function.
Update URL and version in setup.py
Added support for asynchronous execution of Salt commands using the 'yield' keyword.
Fix issue with _dimensions attribute in nc.py
Bump version from 0.7 to 0.9
Fix for handling \r\n in _readline() function.
Fix dangling SocialAccount references by adding a safety check for missing provider accounts.
Added long description and content type to setup.py
Skip test_ipv6_websocket_connection due to Travis errors when swapping between IPV 4 & 6
Added support for custom process identifier in MultiProcessValue
Modified ParseResults class to use next() instead of iterfind() for finding elements in parent.
Added randomization to test file readpartial in TestJFS.
Update print_figure_kwargs to use bbox_inches from environment variable SPY_BBOX_INCHES_O.
Update policy identifier and notice reference organization in PolicyInformation class.
Ensure label points are within image bounds, and update dirty flag accordingly.
Fix for invalid response object in write_response method
Update ClearFuncs to use new keyword argument syntax
Renamed GccArmBuilder to Gcc_armBuilder and updated path variables accordingly.
Added typing to install_requires in setup.py
Added support for specifying a custom staging S3 bucket for batch jobs.
Enable Python subprocesses to work with expect functionality.
Added support for Hive datasets with custom separator character (0x01) in test_parse_many_cases.py
Rounded bar sizes and fixed color handling in Chart class.
Update Requests class to use http proxy instead of verify parameter
Added calculate_markov_matrix method to LexRank class.
Added 'bits' slot to StateVector class
Bump version from 1.2.0 to 1.3.0 and update dependencies.
Added error handling for send_message_chat command in EjabberdctlBackend
Update version number to 0.13.0 and remove develop suffix.
Added include_package_data=True to setup.py
Bump version from 1.0 to 1.1.dev0 and remove release date.
Update CASClientWithSAMLV1 to inherit from CASClientV2 and add verify_ticket method with keyword arguments.
Fixed duplicate edges in graph construction.
Added register=True option to Defaults class and get_skipregister function to check if register command is disabled for a given server.
Fix issue with WsgiServerTransport handling POST requests.
Added support for parsing multipoint geometries with nested parentheses.
Added support for Philips SML003 and SML004 motion sensors.
Added unit tests for Canari project.
Added RemoveAdapterWithDevices method to Bluez5 mock object
Modified BANE.py to remove spurious dimensions from image arrays and added a check for CDELT1, CD1_1, etc.
Bump version from 0.2.4 to 0.2.5.
Modified ClearlyServer class to use Celery instead of celery.app and updated constructor arguments.
Added support for random SP selection in spatial pooling region.
Add caching for job store instance to prevent unnecessary instantiation
Update README.rst with new version number and RtD links
Added import statements for capnp and updated SpatialPooler class to use capnp schema.
Add xfail marker for cross-engine read/write NetCDF3 test due to issue #2050
Update EC2 instance types in test_sles_ec2_network.py
Removed version number from hosts.allow configuration
Fix for #12345: Implemented _check_if_idle function to correctly detect IDLE support for IMAP connections.
Added support for asynchronous iteration in AbstractIterator by changing __aiter__ to an asynchronous function.
Update copyright year to {creation_year} - {current_year} and update project name to Adafruit MCP230xx Library
Added delete_file and delete_dir functions to clean up assets and directories
Added CI support for test_led.py
InfluxDBDriver: updated wait_for_line parameter in _exec method to use self.port instead of hardcoded 2
Added str() conversion for matplotlib version 1.1.1 in Trajectory.plot method.
Update version number to 0.2
Removed unnecessary print statements in ValidatorService.send_txn
Added method replace_secondary_ids to ReStructuredTextParser to replace alias IDs used in references with the corresponding primary ID.
Fix issue with data_uri filter in templatetags.py
Added support for decoding strings as UTF-8 when casting to Unicode
Renamed method 'hello' to 'hello' with self parameter in class ITestInterface.
Fixed format function in bfe_fulltext.py to correctly display URLs with icons.
Added __repr__ method to _TdsSession and _TdsSocket classes.
Fixed bug in Client class where nodelay was set incorrectly for non-UDP connections.
Added support for FAT file system in parted_partition module.
Update setup.py to use setuptools instead of distutils
Removed unused import: sharepa.helpers
Added support for saving form data to the document in DocumentDetailFormView.
Fix for bit_length function in util.py
Modified get_cli_parser() function in cli.py to update help text for -v option
Update forest.py to use sorted edges in graph_as_dot function.
Added support for parsing traceback and message in BaseBackend.
Fixed encoding issue in ModulesEnv.py by changing `shell=True` to `decode('utf-8').strip()` when checking for the `modulecmd` executable.
Removed unnecessary dependency on msvcr120.dll
Removed get() method and replaced with iget() for better performance.
Added import of 'ImproperlyConfigured' exception from Django and updated the error message for missing Pygments library to use the new exception type.
Added terraform to tests_requirements in setup.py
Added support for Python 2.7 and updated version requirements.
Modified setup.py to update exclusion list for Python 3.0 and later versions.
Removed version setting for dataset file and added logging for no version found.
Added on_leTitle_textEdited slot to update keywords list when title is changed by user.
Added Pipfile.lock to list of lock files in is_lock_file function.
Added flag VERIFY_NULLFAIL to check_signature.py
Added support for passthrough arguments in IceCreamDebugger.format() method.
Renamed `dewpt` parameter to `dewpoint` in function relative_humidity_from_dewpoint.
Added bounds for energy_lo, temp, and chem_potential parameters in uibei function.
Fixed getGeneHomolog function to return the first hit instead of all hits, and removed unnecessary sorting.
Replace fa.terminal icon with spyder.python-logo-up and spyder.python-logo-down icons in console mode.
Update setup.py to include python_requires >= 3.4
Renamed function 'model_precision_recall_curve' to 'model_prerec_and_roc' and updated its docstring.
Fix SSL certificate and key validation in webserver function.
Added exception handling for BadStatusLine in LimitedRequestQueueTests
Ensure SYN_DIR is available before opening URL.
Update TreeTime class to use new inference method for ancestral sequences and prune short branches.
Reduced buffer length from 100 to 20 and changed object creation logic in test_object_spilling.py.
Added support for friendly encoding in pageinfo.py
Updated settings.py with new Celery result expiration time and OAuth2 provider token expiration time.
Added retry mechanism for S3 Azure Blob bypass and fixed URL format for certificate errors.
Bump version number to 0.1.2.
ASGIApp: updated stream_callback to handle StreamingHTTPResponse and removed unnecessary chunked header.
Update _results to self.results in ProblemSpec class.
Update Tika version from 1.19 to 1.22.
Added __del__ method to XidDevice class to properly close the connection and delete the object when it goes out of scope.
Update print statement for better compatibility.
Update test_fileserver.py to use support.unit instead of salttesting.
Update BBCiPlayer plugin to use appStoreState instead of episode
ElementPlot: Remove unnecessary 'legend_field' key from mapping when 'legend_label' is present.
Update version number to 1.3.12.
Added shard index parameter to DistributedSession constructor
Update SAML_CONFIG to use config.get('saml2', 'key_file') and config.get('saml2', 'cert_file') instead of hardcoded paths.
Fix for ConnectionDeadException
Update feed processor to truncate title at 200 characters
Added experimental support for splitting sites into tiles for GEM users.
Added utils import and SplitOutput function from auto_ml library, and modified code to use it for splitting data into training and testing sets.
Update setup.py to use find_packages and exclude test files
Added logging statement for virtual environment check and updated test to use logging.warn instead of asserting.
Added missing register warning with address and size
Update test_tif2geojson.py to use 'in' and 'not in' instead of 'assertIn' and 'assertNotIn'.
Modified payloadBuilder method in MYQL class to accept optional format parameter and default to json if not provided.
Added check to skip purging if no packages need to be removed
Bump version from 0.9.24 to 0.9.25.
Added parameter 'func_name' to add_line_breakpoint function in Jinja2Debug.py
Fix conformance test for pygeoapi
Added proxy support to Exchange class by adding proxies parameter to request method and updated session.request call to include proxies.
Bump version number to 0.7.1.
Reverse PEP and KO orders in Pairtrade algorithm.
Added count attribute and average_roundoff property to RounderBase class.
Disallow whole-archive imports in MSVC linker.
Added check for missing 'To' element in WS-Discovery probe match message parsing.
Update psycopg2 version to >=2.3.1
Added import statement for errno module and updated chmod function to handle permissions correctly.
Added __cache attribute to _Props class to cache the properties object.
Update version number to 1.6.1
Added import of sphinx.cmd.build and updated documentation build logic.
Added requests library and updated wait conditions in acceptance tests.
Update optimization parameters for MRD dimensionality reduction simulation.
Fixed quote function in findimports.py to properly escape special characters.
Update sponsors query to use select_related instead of prefetch_related
Added tests for device get device properties and inheritance
Added long description to README.md and updated version number in setup.py
Added type hints for function parameters in test_parsimonious.py
Fix typo in test_dictregister.py: Change 'ValueError' to 'TypeError' in two places.
Improve performance of TokenList.remove() method by using a list comprehension instead of LOOP(list.remove())
Rename and describe single supplementary zip file for PoA.
Removed unnecessary sleep statement and removed download_remote_reports, download_unittest_reports, and download_coverage_report as they are not needed for testing.
Added logging for command errors
Fix for sending file with correct path separators in PyPI vendor dir.
Added support for Python 1.27.1 in client_matrix.py
Update download URL for PGPy
Added import statement for hashlib and added a new property get_atom_id to the Export model, which generates an atom id based on the slug, max_visible_items, and photo_format. Also updated the url method to use the new property.
Update minimum_needs_postprocessor.py to use decimal value for evacuation percentage instead of integer.
Update version number to 0.8.15.
Bump version number to 0.7.0.
Added assertions for job_id in JobTestCase
Removed unconfigured settings from FlaskLoader
Bump version number from 3.7.0 to 3.8.0.dev
Replace instruction argument string with hash value for better uniqueness.
Update base URL of OpenStreetMap API from http to https
Update version number to 1.4.11 and update copyright year.
Fixed sleep function in Bittrex class to use call_rate variable instead of hardcoded value.
Update namedb.py to use epoch-based namespace lifetime multiplier for preorder checks
Added cumulative returns since policy update for PPO trainer.
Added support for freezing columns with all cells evaluated.
Replace _embed_code_links with a version that uses full_function_name instead of cobj['module'] and cobj['name'] separately.
Added support for Osram Flex RGBW LED strip and outdoor version.
Update version number to 0.2.1b2 and add support for Python 2.6 and 3.0 in setup.py
Added support for 3 Hz channels in FIL_reader.read_all() method.
Added column width and format options for numeric columns in SheetFreqTable.
Added get_autosync_payee method to LedgerPython and HLedger classes.
Waveforms.py: Changed condition for drawing waveforms from 'self.n_spikes > 0' to 'self.n_spikes is not None and self.n_spikes > 0'.
Updated configuration for schema external store
Removed environment variable OQ_ENGINE_MODE
Rename test method from test_raises_when_zeep_web_fault to test_raises_when_zeep_exception
Update SSH wrapper state.py to use _salt_.kwargs instead of **kwargs
Added SupportedServices.register_resource_view(queryset.model, resource_view) to register resource views with the queryset's model.
Fixed download_lectures function to use 'with' statement instead of 'open' and 'close' for writing m3u file.
Modified EClass.py to replace metaclass with _metainstance and fixed typo in __getattribute__.
Added support for getting system unit string from base units.
Update binary_heap.py to use 'smaller than' instead of 'smaller its children'.
Modified experiment.py to use os.path.join for constructing file paths instead of concatenating strings.
Update VCR configuration to use sets for hosts_to_ignore and update filter_functions.
Removed TODO comment and updated shebang line.
Exclude tests and tasks from package list
Update _APITOOLS_VERSION from 0.5.9 to 0.5.10.
Added TestReactorMixin to EndpointMixin
Fix for negative scenario ID, also added check for int type
Added gradient support for scipy.linalg.sqrtm and updated solve_triangular gradient.
Update data decode tests to use utf-8 encoding and handle non-string, non-data-structure items correctly.
Fixed ReplicaSetConnection to always have slave_okay=True and removed unnecessary property assignment.
Modified Process class to use new create method with command and attach parameters, and removed quantity parameter.
Fix http://crbug.com/394454 on cros by disabling testIFrame on windows.
Update version number to 0.6.0.dev20191212
Fixed bug in BoundingBox.get_overlap_slices: raised ValueError for negative ixmin or iymin
Added test for output format inclusion
Update py26compat.py to support Python 2.7+
Update version number to 0.3.2.
EditorStack: Update important extra selections and highlighting
Update EMS API client initialization to use a custom transport class and enable caching.
Added deepcopy import and modified createTree function to use deepcopy for caching.
Replace StepLR with MultiStepLR in train.py
Update instrument configuration handling in mode_run_common_obs
Allow controller upgrade and give second bootstrap manager a unique name.
Added || true to pip_packages and changed ls -h to tree --noreport -L in git_repos
Added aspect ratio calculation to Beam class.
Added PIL warning removal to cleanup function.
Removed unused function mock_raw_event and simplified alternate function.
Added labels and help texts to AssignShiftForm
Removed unnecessary code in LlandControl.py
Added check for calc_id in datastore before querying database
Update AMIProtocol to correctly detect protocol version
workerpool.py: Send number of workers instead of length of workers list.
Removed unimplemented purge_db method and updated no_such_entity message
Added 'pylab' to MOCK_MODULES in doc/conf.py
Added import statement for fgivenx and added skipIf clause for test_param_logx_diagram and test_bs_param_dists to run only if fgivenx is installed.
Added support for Kuijken & Gilmore (1989) potential in KGPotential class.
Changed InfoFmtString from '...%(message)s' to '-> %(message)s' in kappa/context.py
Added textupdate widget to AsynOutportPart's create_tags method.
Set hash for Pipfile.lock only if it's not None
Added connection_lost method to StunProtocol class and removed unnecessary socket error handling.
Update _precache_zipimporters to return a dict of paths to zipimporter instances instead of creating new instances for each path.
Update on_teams_members_removed method to include team info parameter.
Added error handling for starting worker in beam boot
Updated KaplanMeierFitter class to use exponential Greenwood formula for confidence intervals instead of the previous method referenced in a PDF document.
Increased maximum number of results for geohash lookups from 10 to 20.
Added support for infinite expressions by importing math and using it to check for infiniteness in the add and mul methods.
Added try-except block to handle errors when creating directory paths in verify_env function.
Added newline character to 'task' field in NIDMResults constructor for better readability.
Modify _toggle_term_protect function to use get_node instead of LocalClient
Added child watcher for asyncio event loop
Improved brain.py by updating the scoring system to use a negative score for unknown answers and added a check to prevent similar answers.
Update to_https_protocol to support tigerdata protocol and use cloudfiles.paths.to_https_protocol instead of manual replacements.
Upgrade upgrade_master.py to use configurable configuration file name.
Added check for zero target packages and updated pytest arguments
Added URL to setup.py
Replace 'range' with 'xrange' in pagerank_weighted.py
Normalize model output and separate output values for Sobol analysis.
Fix typo in release_resources method name.
Added '🕹️' and '🎞️' to double_width_emojis list in rich_helper.py.
Added support for parsing GameEntity nodes in PowerLog files.
Update setup.py to use distutils.core for setup
Added support for libxslt in the FreeBSD deployment script.
Removed unnecessary named capture groups in split_address and split_housenumber functions.
Added support for remote images in RinohBuilder
Fix parsing of SQLite URLs in test_dj_database_url.py
Fixed taxon names with backslashes to work around MultiQC issue.
Added 'has_role' method to PersonEntity, and updated 'role_in' method to use 'has_role' instead of 'members_role'. Also fixed a bug in 'value_from_partner' method.
Update to use getfullargspec in Python 2 and 3
Update MachineCategory model to set _datastore to None by default and only set it to the datastore if one is provided
