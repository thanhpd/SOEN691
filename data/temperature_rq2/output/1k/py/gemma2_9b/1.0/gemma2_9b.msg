Detach chunk adapter buffer and disconnect NodeMap on harvester shutdown
Update S3Manager to use list_buckets method
Refactor error reporting logic for better clarity and efficiency.\n\nChanges:\n- Simplified conditional checks for EC2 instance.\n- Improved logging messages for clarity.
Profile server loop execution with cProfile

Handle different versions of Python's `collections.abc` module consistently.
Simplify store.removeAssociation test assertion
Fix: Correct placeholder in file content comment
Remove debugging print statements from `unescape` function
Add traceback printing for failed module loading.

Update paths and timespan for vb suite
Add option to disable automatic cross-seeding and explicit label for it\n\n - Add `--no-cross-seed` option to prevent automatic addition of a cross-seeding field to the info dict.\n - Change help text of `--cross-seed` to be more specific.

Fix renormalization handling in spparser
Refactor: Handle file descriptors more efficiently in `get_file_chunk_consumer` method.
Add minimum BPDU length and register type
Enable digital pin reporting on port initialization.
Add newline at end of file
Remove redundant `legislative_input` initialization and default value.\n\nA class variable can now be set directly with its value in the initializer without requiring a separate flag.\n
Handle potential errors when retrieving vault information from Braintree
Refine GO DAG version string formatting for clarity.
Add color definitions and resource link.
Use classmethod for b58decode function in Address class
Fix: Handle empty YAML blocks in doc metadata
Refactor: Remove unnecessary print statement and assertions.
Fix: Add back last_modified parameter docstring for send_file function
Change heartbeat logging from exception to debug

Add probe service and constants
Add group filtering to `list_metafeatures`
Fix: Correct regex pattern for CRTF header identification


Add 'release' as a trigger for building Sphinx docs.

Add Darwin platform support
Ensure global monitoring manager endpoint is registered before interaction
Bump version to 4.0.2

Add tests for index dot slices
Move `SoS_Exporter` definition to .converter
Add default configuration files for broker and client
Add flake8 ignore comment
Set MACOSX_DEPLOYMENT_TARGET for OSX builds
Add optional type hint for `window_shell` argument in `Session()` constructor
Update wheel dependency to 0.30
Use 'Agg' backend for matplotlib in syncpr tests
Fix: Avoid resetting parsed streams unnecessarily (#123)\n\nThe code was resetting parsed streams even when it wasn't necessary. This can lead to redundant parsing and potential performance issues.\n\nA TODO comment has been added to highlight the area that requires further investigation.
Add support for AWS profiles
Add `captureoutput` tag for capturing template output into a variable
Disable closing memcached connection on teardown
Bump version to 1.4.0.dev (remove 'dev' from __version__) }
Add blur option to heatmap
Add test for code block parsing.
Add Mizuki voice to Polly resources

Use delayed execution for NinjoTIFF writer
Refactor: Simplify Git Formula cloning logic by removing unnecessary check for existing repository.
Increase thumbnail size
Refine engine output handling\n\nFilter out comment lines from unexpected engine output.
Refactor: Add bool_only parameter for numeric operations across columns
Update Action.to_rest() to accept dp object
Remove retry logic from k8s tests

Avoid reading read_only fields for non-existent resources
Add root_url parameter to JSHost
Simplify `open` method by using **kwargs for file opening parameters.
Fix: Use partials argument in Renderer initialization.

Refactor: Use Connection class for accepted clients.
Handle property definitions in _impl phase
Add support for ellipse regions in ShapeList output
Add assertion for empty response content in HeadTestCase.
Refactor: Improve handling of VCF sample name updates during varscan processing.
Pass config to message processor constructor when available
Simplify boolean_and_simplifier by handling cases with a single argument.
Fix circular import and type hint board in Session class.
Handle potential TypeError when parsing temporal attributes
Refine source path cleaning for consistency


Allow specifying number of trainable ResNet layers in `resnet_fpn_backbone` function.
Handle datetime objects correctly in `is_working_day`
Add test for reading empty arrays with various dtypes (#41241)
Fix permission check logic for yAuth

Handle selection of whole columns and rows in ArrayEditor
Skip test_fuse_projections due to [Reason]
Add new StageDescription properties for API Gateway stages
Refactor: Remove unnecessary print statements from SqlaTable
Add method name shortcuts for call_ and call_mut_\n\nA shortcut has been added for `call_` and `call_mut_` methods allowing you to directly specify the function name after the method call. Example:\n```python\n lens(['alpha', 'beta', 'gamma'])[2].call_upper()\n```\n
Only support .rst files for documentation
Refactor: Use droplet_uids instead of droplet_ids in job execution
Handle potential OSError when getting file size
Refactor test names for consistency.
Add test for sstable_size_in_mb option
Increase iterations for benchmarks
Fix: Handle integer and binary variables correctly in GLPK integration
Fix: Skip doctest for `normalize_arxiv_category` function
Fix handle version increment in reversion signals
Set _highest_parent_ for all parameters
Add script for starting validator with delay
Fix typo in HPC configuration template
Refactor: Plot potentials instead of densities.
Remove unnecessary exception handling in pkgrepo state
Update url patterns to use list syntax
Fix HTML parsing by removing unnecessary boolean check.

Bump version to 0.3.3 and update development status to Production/Stable

Use pdfrw instead of pypdf3 for upscaling PDFs
Add project metadata and classifiers
Bump version to 1.8.6
Add debug logging for SRPM resolution
Bump version to 0.3.1
Refactor: Simplify append logic in file state

Use run_sql for bibwords search query.
Enhance HolodeckEnvironment tick limit reporting
Clarify behavior of pytest.approx() and compare it to other comparison methods.
Use ssh_user for tty default value.
Enable stop button and set algorithm stop flag on execution.

Fix country flag image replacement regex for consistency (#XXX)

Add a new helper function 'foo' to Motor class.
Fix source detachment handling in Stripe model (#1234)
Fix: Use not keyword instead of empty check for block_signing_key
Fix PEP517 build backend handling on Python 2
Add __hash__ method to ContextOptions (temporary)
Refactor: Handle file cache retrieval and closure.
Add network variable to etrago function
Use explicit path for git config file
Check for existence of install config file before loading
Rename ASN1_raw to ASN1_force and handle nested ASN1_Objects correctly
Fix: Call init_app on LoginManager instance
Add charset-normalizer as a dependency

Add Thermo-specific metadata parsing for isolation window m/z
Use DateTime(timezone=False) for created_at columns
Fix JSON generation for component parameters
Fix: Use underscore in local variables names

Resolve issue #88: Set TensorFlow session for KerasRunner
Handle duplicate waypoints and improve logging
Fix: Format invoice ledger lines datetime correctly\n\nEnsure 'From' and 'To' dates in invoice ledger lines are formatted as UTC ISO 8601 timestamps.
Exclude MixtureSameFamily from PYTREE_BLOCKLIST due to performance concerns (#170871051).
Refactor: Use requests for schneier command and modern HTML parsing
Add support for parsing Googlebot, MSNBot, Yahoo!, and Ask Jeeves user agents.

Fix current electrode ordering in EIT40 importer
Fix: Use ReceiptPDFView for download name test
Warn if name, email and username are missing during enrichment
Add special login handler for Arista devices
Refactor: Use `significant_figures` instead of `float_prec` for CIF file formatting.
Bump version to 2.0.0
Refactor status.pid test for better accuracy on windows
Refactor: Find nearest expected polar angle for brute-force optimization\n\nThis commit introduces a helper function `find_near_pol_angle` to efficiently determine the nearest expected polar angle given calculated empirical values.  The function adjusts the range of calculated angles to match the expected range (0 to 2*pi) and uses Euclidean distance to find the nearest match. This improves the accuracy of the brute-force optimization process by reducing the search space.
Clarify environment variable usage in BenchExec tools
Add 'wafer.compare' app to INSTALLED_APPS
Refactor: Update prep_bootstrap to return tuple
Fix: Pass includeDescription and includeDeprecated as non-nullable arguments in IntrospectionQuery.
Yield full hit document instead of just '_source' when retrieving results from Elasticsearch

Add check for image existence before using custom URL prefix
Add color indexing and support for getting RGB values using indexing
Use sharedFileName directly in UUID generation for shared file IDs.
Improve timeout handling in HardwareSource tests
Use f-strings for error messages in int_to_roman function.
Use timeout from Context object in Executor's execute_action method.
Bump setuptools requirement to >= 3.7.0
Support parsing 'i' specifiers along with 'd', 'u', and 'x'.\n\nThis commit addresses a limitation in format string parsing by ensuring that the `_sim_atoi_inner` function now correctly handles both 'd' and 'i' specifiers, providing a more comprehensive solution for integer format interpretation.
Bump version to 0.8.1

Use _lattice attributes for primitive structure comparison
Fix: Clarify literalinclude usage for example DAGs
Fix imports order in admin_interface_tags.py


Update BioGrid dataset versioning to use URL and ingested file number.
Refactor: Remove clock difference tolerance check from client logic.
Print test statistics after assertions
Fix: Use '>' for self.per_experiment_nb_cpus comparison instead of len()
Remove unused `call` function from subprocess module imports.
Improve performance test by adjusting entity creation range
Refactor: Check for valid JSON only if value is a string

Fix: Link loop expression to condition instead of startLoop


Use README.rst instead of README for long description
Retry connecting to USB mux device on failure.
Update `Mapping` import to use `six.moves.collections_abc`
Bump version to 0.0.6
Fix include file handling by concatenating include directive with newline
Bump version to 0.14.1
Remove matplotlib dependency for tests
Only include environment pairs with '=' in singularity.py parser

Relaxed PVD unused field validation
Improve PluginHub.__dir__ to include active plugins from presets
Add Luciddream build type
Bump version to 0.1.17

Ensure log file is writable before using it.
Refactor: Remove redundant tearDown method in TestSetupCLI
Use modern StringIO implementation
Fix initialization based on QT_API version.\n\nThe initialization of OffsetCalculatorMode now uses a more robust check based on the environment variable `QT_API` instead of simply looking for '4'. This ensures compatibility with both PyQt4 and PySide APIs.
Bump version to 0.0.7
Fix: Game status callback type mismatch
Use sets for checking unique keys in probability map
Refactor: Simplify blocking check in TurkeyExperiment.
Import boltons.funcutils for wraps function.
Bump wcxf dependency to >=1.4.6
Renamed `costType` attribute in ChampionSpellData to `cost_type`.
Clarify help text for --absolute-time option in globus endpoint is-activated.
Add retrieval and printing of SVN repository properties
Fix: Pass station ID to get_ad_metadata() call in get_ad_item()

Ensure sigma is non-negative in NormalMessage constructor
Add plotting functionality for qubit layout.
Improve Python import order in conf.py
Ensure integer dtype for random data in merge tests
Change log levels for debug information
Fix GPS heading calculation in FlightGear integration
Refactor: Move vtk type mapping functions to meshio._vtk_common
Fix: Improve logging messages for GenericGeometry.


Bump version to 0.0.0
Fix JPEG encoding for chunks with multiple bands
Simplify package structure in setup.py
Fix: Handle duplicate sample names in verifybamid module
Fix language handling in page index preparation
Fix: Properly unregister actions from child controllers when they are destroyed.
Calculate tax rate for InvoiceItems excluding zero total values
Fix Image model size calculation by handling empty img data.
Add security warnings about running testapp in production
Fix: Use '-c' instead of '-C' for salt-cloud execution
Implement order retrieval, withdraw and deposit address functionality.
Improve command execution by storing last stdout and stderr for easier access and debugging.
Add edx package to setup.py



Add tests for bounding box overlap slices and shape.
Refactor: Use f-strings for formatting well information in logging messages.
Refactor: Release file lock in a safer manner.
Ensure at least one task is defined for builds.

Convert non-followers set to list before iteration
Fix path generation and instance lookup in `view` class.
Add support for filter parameters in CouchDBRiver
Bump version to 0.2.1
Bump version to 2.1.3
Add error handling for ProgrammingErrors during queries.
Refactor: Simplify encoding logic for task
Bump version to 0.1.0a2
Add OCSP key generation and CRL caching
Execute the state machine until a specific state. This state won't be executed. This is an asynchronous task
Fix test_fork_exec by using sleep 0 instead of sleep 1.
Refactor: Rename `vertical_effective_stress` and add deprecation notice.
Refactor: Replace Timeout exception with custom class

Allow empty SQL passwords for read-only access.
Add test for creating containers with CPU RT options (#xxxx)
Use six.assertRaisesRegex for ValidationService test
Rename Markov class to `Markov` from `Markov.Markov`
Add result property and lazy loading for AsyncResult
Improve performance of hazard curve insertion by using a cache.
Avoid PK-only optimization for nested hyperlinks fields.
Update `OS_TO_CLASSIFIER_TAGS` with 'macOS' instead of 'Mac OS'
Add tests for isinstance() with int conversion.
Update impact file handling to use impactLayer.
Allow specifying exit command for exiting enable mode
Make stream thread daemon status configurable
Improve legend documentation in Axes class.
Refine `simpledeps` test with '-x enum' flag
Add support for 'singlehtml' builder

Add __unicode__ method to KeyPublicationStatus and improve key-api syncing logic
Bump version to 0.3.7
Improve error/warning handling consistency in ROOT integration
Update gateway log messages to include host and port
Fix: Handle invalid command arguments more consistently\n
Fix indentation in get_current_user method
Enable multiple commits for autosynth

Update GIT_ATTRIBUTES for dataset processing
Adjust bar spacing and width for improved chart visualization.
Add backports modules and related package data
Bump version to 1.3.0

Deprecate and remove `duration` attribute from classes.
Disable SSL verification for API calls
Add test for random control frame
Fix: Open output directory using QUrl.fromLocalFile
Update handling of 'use_classrooms' setting in course settings.
Use utf-8 encoding when reading the output notebook file
Add test for creating TorConfig from protocol interface.
Refactor: Rename _update_recurring_next_date method for clarity and accuracy.
Add at least status column if results are empty
Use yield from for recursive file listing

Improve handling of missing grid mappings in xarray dataset parsing by logging warnings.
Refine API documentation for content types.
Fix regex pattern in zypper module
Add ContentItemRelation to ContentsEntryMixin for better admin object discovery.
Refactor: Return `res` instead of `lst` in `get_all_states`
Fix: Display mouse position without offset in screenshot
Refactor: Remove redundant `print_property` method
Add test for SNMP GET failure when env is shutdown and timed operation.
Fix: Pass options to runNpmInstall() correctly


Fix updating particle positions in state update function.

Add map editor functionality
Add __unicode__ method to CssBackground
Add support for 2D WCS coordinate transformations
Remove include_package_data from setup.py
Update HealthEventFilter to include 'upcoming' status
Update Application version list in client_facades.
Refactor: Improve documentation links for SegmentationPurityCoverageFMeasure class

Add `statistics` import to titled imports test
Use OrderedDict for path, object, namespace mappings
Use f-strings for string formatting in ExternalTable class

Ensure CLI scripts are rewritten for each test run
Add Calamari and RadosGW log dirs to ceph plugin backup\n\nAlso include detailed health status and df info in ceph output commands.
Increase retries for DataProc job get call (#5)
Fix: Return symbol from find_market for dicts
Remove unused subHMM models plugin
Update test_magicc_data_append to handle datetime indices
Add test for full path retrieval and number of children for large files.
Bump snorkel-metal dependency to >=0.1.7
Refactor: Use different import paths for get_supported based on pip version.

Add tests for larger numbers in Portuguese (pt_BR)

Allow setting known hosts for default port and ports without ':' in hostname.
Add support for RHEL and Fedora distros
Allow user-defined models in BayesianOptimization
Handle potential TypeError when setting minimum magnitude from input parameters.
Add tests for negative temporal parameters in split function.
Use README.md as long description if pypandoc is missing
Set API version in AppObjBaseClass
Refine test execution by ignoring hgroup and html-its if W3C spec is followed.
Remove default reactions directory and parsing logic
Improve CompositeBase initialization by adding optional default image config argument and handling None values.
Remove header authentication from Twitter API call
Format footer div tag correctly


Refactor: Update impact function names in registry test
Log sensitive data for POST requests with 'XXXXXXXXXX' masking.
Rename `dictionary` to `f2p_dictionary` for consistency
Ensure deleteFromStore respects order and limit with UPDATE_DELETE_LIMIT enabled
Remove ordering from ArticleAdmin

Fix: Use ARGS.outputDir instead of ARGS.output for identifier output.
Refine task execution logging and output for clarity.
Refactor: Use testbed's init_blobstore_stub instead of registering a custom stub

Bump swauth version to 1.0.9dev
Display memory in MiB instead of GB.
Remove Python 2.7 specific dependency
Fix websocket endpoint handling
Refactor deploy task: Add product environment dependency\n\nThe `deploy` task now requires a product environment using @tasks.requires_product_environment. This ensures that the necessary dependencies and context are available before deploying.
Use dedicated function for long description and remove unnecessary codecs dependency
Refactor: Move element_methods to be more efficient and readable
Fix circuit extension command to handle both ID strings and router objects
Add xliff file type support
Refactor: Remove unnecessary keyword arguments from functionTagProcessor constructor.
Remove CancelAndReinstate permission from setuphandlers.
Add os module for file system interaction
Remove unnecessary close call in plotRealImg function
Share input embeddings between encoder and decoder
Update coincidence workflow to use inspiral_outs_dict instead of insp_files_dict.
Refine description of `ref` parameter in Project.trigger_build

Fix: Truncate explanation count in CI env
Fix: Assert `pathtotool` is absolute path only on Unix-like platforms

Handle two-factor authentication during login
Add functionality to create new addresses within a wallet.

Rename '_normalization' parameter to 'normalization'
Restrict interpreter versions tested on MacOS for performance

Fix: Handle empty entity JSON data gracefully.
Remove debug logs for container objects in Kubernetes metrics collection

Fix flash write calculation for different block sizes
Refactor: Simplify formatting of messages in `PercentStyleMultiline`
Remove hidden attributes from CRUDResponse flatten method
Improve Pipe repr by using function name for LambdaComponent
Bump version to 0.21.0
Save memory by shutting down parallel processing after composite source model generation.
Use Dark2 color scale for lineage assignments and adjust color mapping
Fix calculation of total seconds in FixedOffsetTimezone
Fix remove_dir on Windows using temp directory
Upgrade LoginAccount version 1 to 2, handling potential ASCII encoding issues.
Improve endpoint configuration handling.
Handle potential errors during server info fetching, including retry logic and host validation for HTTPS endpoints.

Fix PlayerManager handling of packets with entities missing names
Use config for connection parameters and fix superuser values
Fix ownership of remote repository directory
Improve error handling and logging in `clustmap.py`.
Add support for 'linux' platform in native EOL style mapping
Refactor: Move `verify_certs` function outside class

Add missing elif statement for OMIM phenotype entries.
Fix negative binomial log prob autovectorization bug [b/147743999]
Add wait for ready state complete after sending keys
Add support for 1D input points in CDR
Refactor: Move ping option handling into get_ping_command_list.
Fix: Handle invalid input and reset prompt.
Update test data URL for MDF files.

Refactor: Parse dumped grids as single objects
Refactor: Separate argument handling and field setting.


Fix: Use comparison operator in `push_and_tag_images` call
Update XLM-R tokenization description

Allow filtering of Spectrum objects in-place (closes #issue_number)

Add docstring example for tag_ner function
Add django-oauth-toolkit and clean up build artifacts.

Bump version to 0.1.3
Assert rundir is configured in get_pidpath function
Update M4 lexer name consistency
Refactor: Extract channel name from URL directly instead of relying on HTML scraping

Add support for grouping by multiple columns.
Add connection state tracking for FTP target
Handle potential missing `fastavro` dependency
Add test for unreleased items being placed in 'unreleased' release
Bump version to 0.8.1
Fix typo in `concatenate` method docstring

Add close method to DaskResource and ensure resources are cleaned up.
Add support for Python 3.6 and 3.6 notebooks
Fix import of 'stl' module and handle missing modules
Add as_dict method to ConfigModel for getting a copy of config in dict format.

Refactor: Improve timezone handling in template rendering
Ensure URLs and bodies are encoded as bytestrings for Python versions <3
Update task name for stopping build jobs (#<ticket-number>)

Use functools32.lru_cache for Python 2.7 compatibility.

Add `users` attribute to notification rendering
Refactor: Improve omit function handling for different data types.
Add RegionProposal and ProposalPostProcessor to scala class list
Use 'generate' instead of 'setgenerate' for regtest transactions
Refactor: Move DEBUG setting assignment within handle method
Bump version to 2.3.0.dev0
Add return statement to randomizer_bin_und function

Remove unused unittest imports
Add `line_break_count` parameter to `write_line_list` and use it in  `write_chapter`,  and `write_file`
Import `Path` directly from 'path' module
Improve documentation for HazardMapDBReader and its methods.
Fix: Use bytes for authentication strings
Remove debug logging from urlencoded_processor
Simplify PandasQueryCompilerView constructor and remove redundant mappings.
Refactor: Use renku committer and format date\n
Add more tests for query format and fixed in

Use display layers data index for complex display type chooser
Add `file` and `line` parameters to `customized_warning`

Use unittest.mock or mock depending on environment
Make --datacenter required in iface info command.
Switch from ChromeDownloader to WgetDownloader for urlretrieve
Add tests for DeVosSolarcell behavior with zero bandgap.\n\nTests now cover:\n-  calc_power_density returning unit of W/m2 when bandgap is 0.\n-  calc_power_density returning 0 when bandgap is 0.


Fix HTML encoding detection in minification

Improve MQTT client disconnection by properly closing the socket
Add safe string marking for email preview link in EmailTemplateAdmin
Add positional and keyword arguments to handler registration
Remove object_class from BaseModelResource Meta class
Log Saltfile loading information
Add support for list addition in Force objects.
Refactor: Move view name and description retrieval into `rest_framework.views`  
Implement Hessenberg decomposition via Householder transformations.
Ensure object data is encoded as UTF-8
Fix: Delete domain doesn't remove others & test suite updates
Refactor: Move low-level function lookup to the last step in type conversion
Fix SSLWantReadError handling in HttpUA
Set Windows taskbar icon for conda-manager app
Explicitly set encoding to utf8 for reading locale files on Windows
Refactor: Skip nested fields based on FK presence.

Add per-file output for Pylint reports
Fix: Reverse file path and ID arguments in jobStore.updateFile
Refine listener cancellation logic
Add created_at to StatementAdmin list display and filter
Add error logging when attempting to abort connection without abortConnection method
Add new_random_token function for generating random tokens
Add `cross` parameter to spectrogram method
Add viz module
Add bs4 import for Itella module.
Add more informative error messages for failed D4S2 API requests

Add random hash to minion config file path
Refactor Sentry404CatchMiddleware to improve efficiency.
Test lldp execution with mock_lldp and output
Remove unnecessary print statement for debugging in perturb function
Fix battery alert notification logic
Fix: Generate public key if it doesn't exist
Add get_items method to Cart class
Fix git version parsing and PR regex.
Improve type hinting for `excluded_entities` parameter in `bulk_lookup_rdap` function
Improve efficiency of `get_statement_jsons_from_hashes` by using a subquery for finding hashes in the database.

Use image name instead of 'self.image' when tagging and pushing
Improve agent shutdown by using SIGKILL instead of SIGINT
Set Content-Type header for DELETE requests in AioHttpInterface
Fix regex patterns for notebook address and directory

Add TensorFlow test module.
Refine regex for string quoting in completions



Refactor: Use args instead of arg_list in itertools.product
Fix: Return correct value from load function
Fix memory calculation for LSF batch system
Fix: Remove unnecessary type conversion of serial number in create_crl()
Use `abc.Iterable` instead of `collections.Iterable` for type checking.
Ensure compatibility with Python 3.7 and handle `ExceptionGroup` on non-Python 3.7 versions.
Fix: Handle resolving inexistent relationships correctly\n\nThis commit addresses issue #283 by ensuring that attempts to resolve an inexistent relationship raise a `RelationshipClassNotDefined` exception.\n\nPreviously, trying to resolve a non-existent relationship would lead to unexpected behavior. This fix includes:\n\n* Adding a new test case (`test_resolve_inexistent_relationship`) to verify the functionality.\n* Modifying the test data to simulate an absent relationship type by deleting it from the node class registry.\n* Using `pytest.raises` to assert that the expected exception is raised when attempting to resolve the nonexistent relationship.

Configure logging level for requests in Nova cloud provider.
Fix: Import helperservices
Remove unnecessary Optional from expected_warning type hints
Fix BinaryQuadraticModelStructureError message
Add beautifulsoup4 dependency
Fix potential memory leak by initializing `req`

Update Lambda runtime options to include Python 3.9
Add input functionality for tenant schema management commands
Fix: Use `db.session` instead of `db._local.instance` for resource access.
Disable HTML output in QtConsole\n\nQtConsole doesn't report its line width properly and has issues displaying oversized HTML tables. This change disables HTML output for QtConsole.
Fix theme loading by changing directory appropriately
Require API key during initialization
Fix: Use __path__ for static file access instead of pkg directory.
Improve type handling for serializer fields in OpenAPI documentation
Refresh selected state machine and libraries after closure\n\nThis commit updates the UI when a specific state machine is closed, ensuring that:\n- Selected state machines are properly refreshed.\n- Libraries associated with the closed state machine are cleaned up.\n- A fresh list of libraries is displayed.

Add path normalization when transferring stages between ODBs
Refactor test suite: Use virtualchain and keylib

Add medians calculation method to MultiDict class
Test: Switch from testing Schmidt normality to Schmidt seminormality.
Fix size calculation for fixed-size arrays
Redirect LXC subprocess stderr to stdout
Import statements reorganized for clarity.


Refactor: Remove initialization logic from shell setup
Refactor softmax implementation for improved efficiency
Fix: Use 'is not None' for conditional checks in deci2sexa function
Use json.loads instead of json.load for parsing Wikidata response
Fix issue with setting temperature to 0.\n\nWhen the temperature is set to 0, disable automatic NPsis calculation and set the value to 1.
Add SSL support and change default `all_ok` behavior in interactive server.
Fix task management for DisaggHazardCalculator

Allow `const_module` and `instruments` to accept modules or module types for improved flexibility.
Reduce maximum pool size to 50 from 500
Add test for clearing surroundings
Reduce MAX_NAME_LENGTH for better step names
Reduce wait time for subprocess closure in Spyder restart.
Bump version to 0.4.12
Remove extra newline from parsed HTTP headers.
Refactor: Use XPath for link finding methods\n\nThe `find_link_by_partial_text`, `find_link_by_text` methods now utilize XPaths instead of relying on Selenium's explicit `find_elements` calls. This improves readability and consistency with other finders.
Ensure DictInterface handles non-tuple data properly.\n\nAvoids consuming data of other types that don't match the interface definitions when using an iterator interface. Fixes potential issues with handling data of diverse formats.

Refactor: Rename `id2info` to `id2source` for clarity

Refactor: Use input_processes for validating and running pipeline
Fix: Return explicit error for non-Windows platforms
Add sidebars to Sphinx documentation
Fix regex pattern in oEmbed bootstrap function for better matching
Add internal error handling for fuzzy hashing function
Fix typo: 'operationi' to 'operations'
Add dependencies for map display and flight view
Fix: Ensure sets in DisaggregationBinaryMatrixXMLWriter have at least one element
Update serial dependency to pyserial
Use repr for timezone ids in parametrization\n\nThis change replaces the hard-coded strings with repr(timezone) to make the test output more informative and consistent.
Use direct class instantiation for event handling

Use setUpClass and tearDownClass for API server process lifecycle.
Refactor: Improve readability of Checkpoint instantiation
Use frozenset for storage parameter in get_user_data

Add tests for different branches and fossil installation check
Fix autoname for Viaplay clips and episodes
Refactor InterpolationTask and LinearInterpolation: Standardize library parameter

Modify `isChanged` function for more accurate change detection
Add DocBookBodySubNode
Ensure transferred files are gzipped.
Save sent emails to database instead of sending them.  \nReturn number of messages sent.
Add options for Map and HUD in console menu

Refactor: Test for getting a resource list

Add axis labels and grid to timeseries plot
Update database connection string
Use getattr to retrieve fill_value from NDArrayField instances

Remove debug logs for House Status Monitor Enable/Disable events.
Remove unused urllib.parse import
Fix Docker connection handling in glances_docker plugin
Remove HipChat provider

Fix nested value lookup in YamlTestCase
Print call arguments and error output instead of logging
Add focus to canvas after sending command
Add support for importing numpy conditionally and using contextlib

Use a set for WEBHOOK_EVENT_TYPES
Fix: Remove trailing '*' from translated protein sequence.
Use all available CPU cores for multiprocessing in `scan_dir`
Change tooltip background color to configurable value\n  TOOLTIP_BACKGROUND_COLOR is now a constant and used to set the background color of tooltips

Remove rewriting of maven_metadata_path
Permit certain punctuation characters within tokens
Fix: Handle `action` argument as boolean instead of False value
Add models_dir function
Lowercase package directory name for consistency
Handle return value change in find_and_modify for MongoDB >= 2.1.2 (#SERVER-6226)
Clarify UnknownExtension exception message
Add `self.choices = list(self.choices)` to ensure it is mutable and avoid errors
Add --websocket-port only if --connect-existing is not specified.
Add functionality to OrderableManager for accessing previous ordered object
Refactor: Remove support for Python 2.6 BZ2 handling
Add mkdir patching for tests_blog.
Fix return data structure consistency for calls to local and publish commands.
Fix: Handle `None` result correctly in test suite.
Refactor: Use py_modules instead of find_packages
Remove strict validation from RequestSchema in tests
Update dependencies: Remove moto version pin and botocore pin.

Fix: Ensure `_git` instance is modified after baking for non-TTY output.
Add CRUD event creation for M2M changes
Improve efficiency of composition comparison operations.
Fix: Properly format public key for verification
Fix: Use sampling with replacement for all parameters when distributions are present in RandomizedSearchCV
Add error handling for undefined terminals in grammar definitions.
Improve file export output for zip files
Fix: Use Unicode escape sequences for newline and control characters
Use `items()` instead of `iteritems()` for Python 3 compatibility

Configure Red Hat policy for Dropbox uploads
Refactor: Improve server list handling in `dns_exists`
Reset task_try_count on new Task instantiation

Use broker connection URI for conninfo display.
Refine stats display logic
Update model step function to handle state and dones
Fix handling of environment variables and paths on Windows
Enforce -t/--time as a required argument and fix bug description formatting
Skip process exit tests on non-POSIX systems and Python 3
Remove 'index' column from aggregated dataframe for improved test stability.\n\nThis change removes the 'index' column from the aggregated dataframes in `aggregate_losses` and `aggreg` functions to ensure consistency across tests. Specifically, it addresses an issue observed in `test_case_miriam` where the presence of the 'index' column caused unexpected behavior.
Add .gitignore for s3conf directory
Add support for light font weight and refine bold text condition.
Bump version to 2.2.1

Fix CPU frequency reporting on BeagleBone\n\nThe script now correctly retrieves and reports the actual CPU frequency on BeagleBone devices.
Update ARAnalysesField to incorporate new form values for selected analyses.
Refactor: Open HID device using path instead of direct device instantiation.
Add test for argument parser and version check
Fix endianness in FieldLenField for 'len' in RadioTap layer
Add support for synchronous query execution
Add dependencies ujson and sh
Fix FileSystemContextMenu to handle different desktop file extensions (#<ISSUE_NUMBER>)
Add support for shared memory in StimelaJob
Bump version to 0.5.0
Refactor: Use parsing function for samples in ZigBee device
Refactor: Use Infomap instance methods for network operations and output.
Refactor: Remove unnecessary `P_slaac_preferred` parameter from  `configBorderRouter`
Use regex for index redirect url
Fix: Ensure gather operations with list-of-traces only copy once per trace
Ensure local and remote paths end with '/' for consistent routing.
Fix calendar roundtrip issue and skip if numpy >= 1.16 due to pytables build dependency.
Use explicit import for lsb module in hosts.py
Add tests for quoting and escapechar options.
Display memory data as machine words by default
Fix issue with potentially oversized POEs during disaggregation.\n\nThe commit addresses a potential issue where calculated Peak Over Exceedance (PoE) values could become excessively large during the disaggregation process. This has been addressed by implementing checks to ensure that PoE values remain within reasonable bounds, preventing inaccurate results and improving the overall reliability of the disaggregation calculation.
Fix: Improve regex matching for invalid responses in SSH autodetect.
Fix: Properly destroy GoToLineDialog's C++ object when closed.

Simplify modify_parameters method by removing type hint
Ensure images are in RGB mode for locateAll function
Bump version to 0.7.7rc1
Bump version to 0.8.4
Allow updating array attributes from JS
Add STOP_WORDS as a set

Fix: Ensure deployment configuration values are only added if not None.
Increase POSITIONS_PER_XML from 500 to 1000
Refactor: Simplify directory cleanup logic


Emit clicked signal when toggling active state
Test: Ensure deleting a category deletes all its children.
Bump version to 2.0.4
Refactor: Implement unique value tracking in `rowParse` 
Update jar file pattern in test suite (#1234)
Simplify input path handling in executor.\n\nThe code now directly uses the provided `input_paths` without modifying their paths based on the config directory.
Add version printing and project initialization functionality
Use logger.warning instead of logger.warn
Fix Instance property initialization with has_ref attribute.
Fix: Handle potential ordering issues with ID3.get() \n\nSome keys in EasyID3 can end up in multiple frames, leading to undefined order when retrieved using .get(). This commit sorts the values returned by .get(key) before comparison to ensure accurate testing.
Fix: Add newline at end of file
Bump nionutils dependency to >=0.3.15
Add InputBase class for input parameters definition.

Fix: Update large directory processing logic\n\n- The progress bar and logging now accurately reflect the current working directory during large directory processing.\n- Minor updates to path formatting for consistency.
Update libuv revision to 93877b1
Downgrade numpy dependency to 1.16.4.
Add Apache 2.0 license
Fix count calculation for directories containing downloads directory
Add help subcommand functionality and tests
Convert HTML content to native strings in BuildStep logs.
Add environment argument to apextestsdb_upload

Refactor: Use OQLITEIMPORT instead of OQLITEDEVPACKAGES for importing extra packages.
Use _qsize instead of qsize in ExhaustableQueue repr()
Add CorotatingRotationWrapperPotential and GaussianAmplitudeWrapperPotential
Bump version to 0.13.5
Bump version to 0.9.0
Fix handling of unicode strings in test_models (remove conditional)
Refactor: Use lambda functions for minimum operation in MNNGraph.\n\nThis change simplifies the code by using lambda functions to define the `minimum` operation when dealing with sparse matrices.
Copy handlers and symbols for polygon stream
Refactor: Remove Celery distribution for post-processing tasks
Fix: Handle unsupported looping behavior with WorkflowException instead of NotImplementedError
Fix: Install Prism correctly in tests

Improve template rendering debug log
Update Content-Type header formatting
Update project URL to use sr.ht hosting.
Return empty dictionary instead of raising an exception when connection with master fails.
Fix type check for dimension attributes in nc class
Bump version to 0.9
Refactor: Improve newline handling in _readline() method.
Fix dangling SocialAccount by checking for hasattr before getattr.\n\nA `getattr` call without hasattr check could potentially raise an AttributeError. This fix ensures we only attempt to access fields if they exist on the object.
Add README.md as long description
Skip IPv6 websocket connection test due to Travis errors
Handle multi-process scenarios with custom process identifier function.\n\n- Allow specifying a `process_identifier` function for determining unique process identifiers.\n- Update `__check_for_pid_change` to use the provided `process_identifier` function. \n- Document the requirement for `process_identifier` to return distinct values in different processes.
Fix: Use generator expression for find_in_parent method
Fix partial read test in JFS
Fix: Update inline backend configuration\n
Refine PolicyInformation representation for clarity and consistency.

Add error handling for invalid response types and handle connection loss during response writing.
Refactor: Handle `ret_config` within `kwargs` for clear function loading
Rename `gccarm` to `gcc_arm` in build path and variables
Add typing dependency
Allow specifying staging S3 bucket for CWL jobs.\n\nPreviously, a default S3 bucket was used for CWL jobs. This change allows users to specify a custom staging S3 bucket via the `--staging_s3_bucket` argument.
Enable Python subprocesses to work with expect functionality.
Add support for new separator character 0x01 in sepChangeDict
Round bar and factor sizes for improved visual accuracy

Add `calculate_markov_matrix` method for generating Markov matrix based on similarity threshold and discretization.
Add 'bits' to _print_slots for StateVector class
Bump version to 1.3.0 and add six dependency
Handle send_message_chat errors and use code/output from ctl calls
Bump version to 0.13.0
Include package data in setup.py
Bump version to 1.1.dev0 and clear release date
Refactor: Move CASClientWithSAMLV1 to inherit from CASClientV2

Refactor: Skip registration based on server settings\n\nAdds a `get_skipregister` method to determine if registration is disabled for a given server. This avoids unnecessary register calls when configured by users.
Fix: Remove placeholder response from WsgiServerTransport

Add support for additional Philips motion sensor models
Add unit test runner and import unittest
Add convenience method to remove adapter and associated devices
Fix spurious dimensions in image data loading\n\nFixes an issue where image data loaded by BANE.py contained unnecessary dimensions. The code now resizes the array to keep only dimensions greater than 1, ensuring correct processing.

Refactor: Simplify ClearlyServer initialization and parameter usage.
Improve SPRegion behavior when learning is off
Cache job store instance for promises based on its config. Avoid instantiating it for each promise.
Update README badge and Read the Docs links with latest version
Add capnp serialization support for SpatialPooler
Fix: Handle invalid engine types during dataset loading.\n\nPreviously, attempting to open a dataset with an unsupported engine like 'foobar' would raise a `TypeError`. This change clarifies the error message to inform users that they can only read datasets using supported engines.\n\nAdditionally, mark the cross-engine read/write test as XFAIL due to unresolved issues (https://github.com/pydata/xarray/issues/2050).

Refactor: Remove version banner from hosts.allow content
Use `connection.capabilities` instead of parsing `connection.capability()` result
Make AbstractIterator.__aiter__ async
Update copyright information with creation year and duration
Add file deletion functionality
Add CI integration for LED tests
Use configured port for influxd wait_for_line
Fix matplotlib plotting with version 1.1.1
Bump version to 0.2
Remove debug print statements for processor selection.
Fix: Replace secondary IDs in references with primary IDs\n\nThis change introduces a static method `replace_secondary_ids` that traverses the reStructuredText doctree and replaces all alias IDs used in references with their corresponding primary IDs. This ensures consistent referencing even when using aliases for elements.

Ensure consistent string handling in media file representation. \n\nFixes potential encoding issues by explicitly decoding strings using UTF-8 and handling unicode cases appropriately.
Refactor: Add self parameter to hello method in ITestInterface

Add __repr__ methods for _TdsSession and _TdsSocket
Fix: Ensure TCP_NODELAY is set only for non-Unix and non-UDP sockets
Add support for FAT filesystem type
Use setuptools if available, otherwise fall back on distutils
Remove unused helper functions
Save document data from form
Fix: Use bit_length() instead of bitlength() in safe string function.
Clarify verbosity help text
Sort edges for consistent dot graph output
Format traceback for notifications and handle empty traces. Added error parsing logic.
Fix: Decode output from 'which' command in modules.py

Improve block creation and indexing methods (GH 45678)
Use ImproperlyConfigured instead of Exception for pygments missing import
Add terraform dependency to test requirements
Add Python version requirements
Refactor: Use regex for excluding files during installation.
Fix: Set dataset version by date if not provided and log output
Refactor: Update keyword list when title is changed
Add Pipfile.lock to list of lock file extensions
Add VERIFY_NULLFAIL flag for stricter null-signature checking.
Refactor: Simplify IceCreamDebugger return logic
Refactor: Rename dewpt parameter to dewpoint in relative_humidity_from_dewpoint
Add bounds for energy_lo, temp, and chem_potential parameters in uibei function
Refactor: Retrieve single best BLASTP hit instead of top N.
Update console icon to use Spyder-specific logo

Renamed `model_precision_recall_curve` to `model_prerec_and_roc` for clarity.
Refactor: Simplify SSL certificate and key checks in webserver CLI
Fix overflow connection handling on macOS (#X)
Ensure SYN_DIR is available


Improve handling of image encoding in pageinfo extraction
Refactor: Set CELERY_RESULT_EXPIRES to 6 hours and convert time constants to seconds.
Fix S3 URL format for certificate errors when copying blob
Bump version to 0.1.2
Fix ASGI response handling for `StreamingHTTPResponse` and fix content length calculation
Use `self.results` instead of `self._results` for consistency

Close connection in __del__ method
Refactor: Print server address based on unix_domain setting consistently.
Use new testing utilities from `tests.support.unit`
Refactor: Update BBCiPlayer plugin schema for app store state
Refactor: Remove legend_field from mapping if legend_label is present in properties
Bump version to 1.3.12

Use config file for SAML key and certificate paths
Prevent sending pushes from dead connections (#123)
Limit feed title length to 200 characters

Split training data into train and test sets using random sampling (80/20).
Use find_packages for better package discovery.
Improve test environment checks and add logging for missing virtualenvs
Add address information to warnings when filling registers or memory with unconstrained bytes.
Use `in` for property checks instead of  `assertIn/NotIn`
Make payloadBuilder more flexible by allowing 'format' parameter to default to object format.
Improve pipenv purge command by skipping execution if no packages are found for removal.
Bump version to 0.9.25
Add `func_name` parameter to `add_line_breakpoint` function
Fix: Reduce conformance checks in pygeoapi tests\n\nThe number of conformance checks in test_ogcapi_features_pygeoapi.py has been reduced from 17 to 16.
Add support for proxies in HTTP requests
Bump version to 0.7.1
Reverse order placement for pair trade logic.
Add count and average roundoff error tracking
Refactor: Raise error for whole archives with MSVC linker
Handle missing 'To' element in WS-Discovery probe match messages
Update psycopg2 dependency to >=2.3.1
Fix permissions handling for local remote\n\nEnsure that files and directories are properly protected with read permissions for all users while avoiding unnecessary permission escalation in case of EPERM or EACCES errors.

Bump version to 1.6.1
Fix documentation build by using correct Sphinx import for different versions.
Increase timeout for AllApplicationActive wait condition
Reduce optimization iterations in MRD example.
Fix: Simplify string quoting in `quote` function
Use select_related instead of prefetch_related for sponsors query
Add test for device property retrieval during initialization
Update description and add README as long_description
Add typing hints and update test function signatures
Fix TypeError raised when appending non-dicts and incorrect arguments
Improve performance of TokenList by avoiding O(n^2) operation
Improve handling of single supplementary zip files for PoA articles.\n\n- Rename and describe a single supplementary zip file if present in the PoA section of an article. \n- Remove the previous TODO comment about checking for PoA.
Refactor: Remove unnecessary docstrings
Log command execution results for debugging
Fix: Handle slashes in PyPI release paths
Add v1.27.1 to LANG_RELEASE_MATRIX
Remove download URL from setup.py

Ensure evacuation percentage is a float value

Bump version to 0.7.0



Use hash of memory location instead of string representation for keccak256 call in KECCAC symbol.
Update API base URL to HTTPS
Bump version to 1.4.11
Fix: Adjust sleep duration for rate limiting
Adjust preorder lifetime calculation to account for epoch namespace multipliers
Reset cumulative returns after each policy update.
Fix: Handle TypedExceptionWrapper in freeze column
Improve code linking accuracy by using full function names\n
Update Osram Flex RGBW device description
Bump version to 0.2.1b2 and add Python version requirements for IPython
Fix calculation of number of coarse channels for high-density datasets.\n\nThe previous implementation did not correctly handle cases with a large number of channels (>= 1048576), leading to potential inaccuracies. This commit introduces an adjustment based on the channel count to ensure accurate coarse channel calculation for all dataset densities.
Fix: Ensure column width is set when generating numeric frequency tables
Implement `get_autosync_payee` method for LedgerPython and HLedger.
Fix: Only draw waveforms if spikes exist
Refactor: Simplify share configuration in schema_external.py

Refine test description for zeep exception handling.
Fix kwargs passing in highstate function
Refactor: Register resource views based on queryset model attribute.
Refactor: Simplify M3U playlist creation
Renamed `__metainstance` to `_metainstance` for better convention adherence.

Fix typographical errors in Binary Heap documentation
Refactor: Use os.path.join for filename construction
Use sets for hosts_to_ignore to improve performance and memory usage.
Remove TODO comment about Click and setuptools integration

Update apitools version to 0.5.10
Add TestReactorMixin to EndpointMixin for testing endpoints with reactor interactions
Fix: Ensure scenario ID is valid integer.

Fix data decoding for various encoding situations (Issue #1234)

Refactor: Update Process creation method to use command and attach
Disable IFrame test on Windows (#394454)
Bump version to 0.6.0.dev20191212
Refactor: Raise ValueError instead of returning it from bounding box method
Test output format inclusion across APIs
Fix: Ensure strip_fragment works correctly on Python 2.7 and later
Bump version to 0.3.2
Update editor highlighting on color scheme change\n\nThis update ensures that after changing the color scheme, the current line and cell are properly highlighted by unhighlighting and then re-highlighting them.
Refactor EMS API client initialization for improved code clarity and modularity.
Avoid mutable cache issues when creating new tree structures\n\nThis commit introduces a deepcopy() operation when accessing cached graph data during the creation of new trees. This helps prevent potential issues arising from modifying mutable caches within functions like `creatTree`.

Refactor: Simplify instrument configuration handling in `clirundal.py`
Ensure bootstrap managers have unique names and remove unnecessary os upgrade config.
Improve fabsetup listing and avoid failing on non-existent repos

Remove PIL and datastore warnings from output
Remove mock_raw_event function
Remove labels and help texts from AssignShiftForm
Refactor: Simplify EQI1 parameter trimming logic by removing redundant attribute lookup.

Fix: AMI protocol version detection for case-sensitivity
Use `self.num_workers` instead of `len(self.workers)` for reporting worker count
Remove purge_db method from PostgresManagerMixin
Add matplotlib and its submodules to mock modules list

Improve documentation for KGPotential class by adding formula and a description of parameters.
Refine info log format
Add textupdate widget to AsynOutportPart
Fix: Set Pipfile hash in PipfileLock if not already present
Log connection lost and error received events
Improve docstring clarity for _precache_zipimporters function

Add error handling for starting Beam workers
Refactor: Improve Kaplan-Meier CI Calculation
Ensure geohash results are included even with bucket overflow and center\n\nAdd at least 10 items when adding geohash key results to the bucket.
Handle infinite values in expression operations
Enhance error handling for directory creation during environment verification
Update mapping for NIDMResults with task replacement and hash function.
Refactor: Retrieve instance ID directly from connection object.
Attach child watcher to asyncio event loop
Refine similarity check in Brain.py
Add support for TigerData protocol in `to_https_protocol` function\n
Print the actual config file name instead of hardcoding 'master.cfg'
Ensure tests are run only when packages matching glob are found
Add project URL to setup.py
Update pagerank_weighted function to use xrange instead of range for iterating iterations.
Normalize model output before Sobol analysis
Fix typo: 'relase_resources' changed to 'release_resources'
Add more emojis to fix spacing
Extract game ID from log and set in PowerLogParser.
Handle potential setuptools dependency issues
Add libxslt dependency for FreeBSD deployment
Remove unused functions split_address and split_housenumber
Fix subtitle formatting in Rinoh documents.
Fix: Consistent use of dj_database_url.parse() throughout DatabaseTestSuite.
Fix: Handle backslashes in taxon names
Refactor: Clarify `has_role` method and check for duplicate roles
Use `inspect.getfullargspec` or `inspect.getargspec` depending on Python version
Fix potential `None` reference for `self._datastore`
