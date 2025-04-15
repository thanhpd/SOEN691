Add categorization of errors in ErrorList
Make logger field protected
Log configuration at debug level instead of info.
Add helper methods for getting master and worker addresses
Simplify random number generation in H2TestUtil
Improve PRIMARY_KEY_CONSTRAINT_VIOLATION_MESSAGE pattern for Fongo compatibility.
Add crawler initialization and sleep before extraction
Fix logging of frame length in ReceiverUdpChannelTransport
Refactor: Use add() instead of set() in fromStringList()
FLINK-23139: Handle null stateRegistry in discardState() method.
Temporarily hardcode Rho value for testing purposes in ADMMSolver.
Use thread-local SecurityContextHolder for testing
Add constructor for JedisShardInfo with host, name, port, timeout and weight

Remove OutputSymbol class as it's no longer needed.
Add load/unload/reload functionality to MaterialDataPager
Remove unnecessary @Test annotation from testDowngradeHoldCursorsUnderXa()
Use TransactionSynchronizationRegistry for rollback if available, fallback to user transaction otherwise.
Fix: Ensure non-ignored exceptions are propagated when command overrides default ignore settings.
Improve OHttpResponse handling of content types
Add captions to search dialog fields
Remove comment about potential in-memory database optimization.
Improve readability of comparison output in CgIsaParamTest
Use Hamcrest MatcherAssert instead of Assert for assertions.
Make App class final and add missing `final` keyword in main method arguments.

Update management version numbers to 3.0.0
Skip nested CityObjects during unmarshalling.
Use canonical hostname for local worker address resolution
Improve ClientConnectionsShutdown by using channel.closeFuture() instead of a separate promise.
Use PresentsTestBase instead of PresentsTest for RefTest
Use '|' instead of '|' for channel delimiters in replay URIs
Add verbose mode for debugging
Log level change for VirtualType property removal
Use Class<T> for type casting when mocking in AbstractMockingTestCase
Refactor: Resolve main cube before undeploying.

Ensure extensions are terminated before shutting down threads


Fix typo in test suite name
Fix date format in DurationBusinessCalendarTest
Fix handling of empty reference and alternate alleles in Variant parsing.
Refactor: Use dynamic user name based on system properties and class hash in VerifyBase
Remove unused SpinnerAdapter import
Fix escaping of sequence names for PostgreSQL (#XXX)
Synchronize STIXSchema validation for thread safety.
Add support for Byte type in ColumnValidationReader
Use unique identity for leader election lock.
Fix Scrap view type handling in Recycler
Fix: Remove problematic confirm dialog test case for Chrome.
Use ApplicationClassLoader's class loader for loading system classes
Improve documentation for XMPPConnection#addSyncPacketListener and XMPPConnection#addAsyncPacketListener


Fix escaping of source file name in generated code
Add started() method to check if REPL is running.

Use rebind instead of bind for sessionID and channel instances in JNDI manager.
Fix: Clear current span before starting a new one
Reduce test data fraction in DeepLearningProstateTest
Make ClientFactory public
Fix: Calculate target frames based on desired FPS of 60.
Fix: Convert non-byte attachments to byte array using String.valueOf().


Improve GetSchemaInfoCommand by handling both Node and Relationship entities.
Update $Date$ in CharacterChromosome.java
Fix tick handling in ControllableRunner

Set Tomcat base directory to target/tomcat-<class name>
Fix TestIgnorance logic for handling service versions and listing.

Improve ObjectUnderFileSystem handling of listing chunks\n\nThis change ensures that we only create breadcrumbs for folders and not root directories when retrieving object listings. This avoids unnecessary metadata creation and improves efficiency.
Allow visiting fragment spreads in VariableTypesMatchRule
Increase buffer size for server handshake.
Refactor: Simplify getMaxInitialId method
Improve data caching logic for DeepWater model training.
Fix: Handle GET requests correctly in RequestAuthenticator
Disable Spark UI and run tests locally
Add delay before stopping Instruments for RealDevices
Update JavaFormatterTest with new test data

Refactor: Simplify appendOp logic
Update DruidMaster log messages for a more playful tone

Refactor: Remove deprecated Expect.apply() implementation
Remove unnecessary @Slf4j annotation

Update boolean handling in Proto class for consistent type and value representation.
Use proper closing tag for gwt:property meta tag
Use source info from method node for ASSIGN in ES6 class rewrite



Add methods for ending tags
Add check for Steam running status and expose it publicly
Fix testExceed128Connections and remove old version.
Ensure ProcessInstance retrieval includes signals
Warn instead of debug when multiple kubeconfig files are found.
Remove test for SMILES format
Add support for Conscrypt on Android
Update query handling and add corpus selection check in TigerQueryBuilderCanvas
Use 'this' keyword for getter methods
Update BeansXmlDescriptor to use common FileDescriptor
Remove RapidoidInitializer.initialize() call from Conf constructor initialization.
Remove unused MessageContent class and related code.
Add attrX method for creating AttributeExpression
Update DescribeCommandTagsIntegrationTest to reflect correct tag output.
Remove redundant @Override annotation
Improve efficiency of key decryption by reading only needed data


Fix: Use URI instead of External Form for resolving resource files
Refactor: Access aggregator and collector fragments through core field in AdHocPlannedStatement.
Change log level to INFO from DEBUG
Add connectionId to MpScheduler's scheduleMessage
Fix: Remove unnecessary blank line in Violation constructor
Add test for inserting Person with null key reference
Renamed `getCachingAndUpdatingParserHolder()` to `getCachingAndUpdatingParser()` and marked the old method as deprecated.
Add tax ID information to CustomsDeclaration test data.
Refactor: Update AccessLevel enum values for consistency
Introduce default implementation for IIndentationInformation interface #205
Disable anonymous access on Resource Server
Only set filename if path and file descriptor are not null.



Add tests for Muxl and Demuxl
Update raster layer opacity handling
Fix DockerService.getContainerIpAddress() for non-running hosts
Make Connectivity.create() protected


Refactor: Use BorderLayout for layout in ValidateFrame.
Retry remote requests with a timeout instead of blocking indefinitely.
Fix type mismatch in JavaConstantDynamicTest
Improve error message for Mockito mocking limitations
Refactor: Simplify ProcessInstanceIdCommand interface
Simplify invalid packet counter increment
Fix: Correct row/column index error in Matrix insert/removeColumn
SONARPLUGINS-3097: Handle empty line at end of file in Git blame
Make SimpleReadWriteLock Serializable
Improve LifecycleManagerStoreImpl by simplifying binding and tracking logic.
Use an explicit loop instead of a for-each loop for improved performance

Add serialization check for MojoModel
Add event handlers for join, leave and execute events.
Use username and password for hostbased authentication
Refactor: Simplify history management in SwingTab.
Make Encryption a final class

Fix typo in `browseFrom` method and add deprecation for old spelling

Use HashMap instead of Hashtable for properties cloning
Fix: Correct grammar in warning dialog about non-local database migrations.
Fix handling of categorical features with enum values outside training range.

Use user property for random seed overriden attribute






Improve ResizeSettings validation by using a delta for float comparisons.
Only schedule updates if a memory cache is set
Fix: Ensure header IDs are positive (>=1)
Fix PackrReduce handling of directory input.
Update master thrift shutdown timeout unit to seconds.
Increase parameters for motion estimation in ExampleVideoMosaic.
Add note about missing max request size validation
Fix classpath string splitting on Windows paths
Refactor: Remove date from simulation time in log viewer panel.
Use constant for status code 'Too Many Requests'

Add clearViewCached method to AbsViewHolder
Increase test duration for ArchiveRecordingLoadTest
Add CACHE_MODE_LOCAL flag to WriteCommand invocation context for improved performance.
Add methods for boxing and unboxing int arrays
Use Log.getStackTraceString() instead of throwable.toString() for better stack trace printing.

Update OperaDriver description to reflect its purpose and development.\n\nThe updated description clarifies that OperaDriver is a vendor-supported WebDriver implementation developed by Opera and volunteers. It also highlights its role as part of the Selenium project and its use of the Scope protocol for communication with Opera.

Improve HamcrestCondition description by including the value and matcher description
Revert: Load function library on metadata reload (#ISSUE_NUMBER)
Fix type variable cycle test by using TB annotation instead of TA
Disable expression cache in eager mode
Fix NaN handling in Sum NodeList
Add permissions for cluster config entries
Fix: Ensure ApnsConnectionListener is not null
Use ConcurrentHashMap for tracking finalizable objects.
Refine GaussianProposalDistribution for LinearRegressionTest \n\nReduce proposal distribution variance from 20. to 0.1 in `youCanSpecifyACustomSamplingMethodInsteadOfGradientOptimization` test for better convergence.
Add Transfer-Encoding header for HTTP/2 proxy requests
Use default watermark if previous committed states are not found.
Fix calculation of next timeout for TimerTask.
Add SourceVersion entries for Java 1.7 and 1.8
Add tearDown method to clear ContextHolder in X509ProcessingFilterTests.

Make `CheckRange` more generic by using `? super C` for Comparable types.
Update approvalsRequired in AsyncMultisubprocessTest
Update Metrics imports
Ensure database creation is executed properly


Use 'R' instead of 'H' for two-digit year format in JapaneseCalendar tests.
Use a cached thread pool for asynchronous operations

Reduce assertion percentile for performance tests
Refine terminology expansion logic for handling zero count cases
Fix issue #122: Ensure endDate is midnight for period counting
Fix: Handle chunk ID reversion in GameUpdateTask

Add get/set methods for Vaadin Form in AbstractFormPropertyRenderer
Remove dependency on icu4j

Fix: Skip JPEG markers correctly in RafMetadataReader
Fix typo in SpanBuilder startSpan documentation
Fix: Transient fields are not loaded from disk (#[ticket number])\n\nThis commit addresses an issue where transient fields declared in the schema were not loaded from disk during document reload.\n\nChanges:\n- Added code to explicitly reload documents from the cache and verify the presence of transient fields.\n- Updated assertions to reflect that transient fields are not present after reloading from disk.


Fix: Set result index for tree nodes
Remove unused date format
Fix CLI version mismatch handling (see 6835580)
Refactor: Return input field value if 'value' flag is true

Update ReplaceIfSameOperation to store old value on success.
Increase interactive job wait time from 5 seconds to 10 seconds
Improve documentation for VMath.timesTranspose()
Fix termId calculation in ArchiveConductor

Prevent duplicate project creation during copy operation.
Refactor: Simplify setCoordinates method

Adjust resource calendar view styling and hour height
Make datebox popup modal
Fix: Use eGet with 'resolve' flag for retrieving references in BatchLinkingService
Reduce ping interval for SimpleRoundRobinLBTest
Use an ArrayList instead of Collections.emptyList() for combined artifacts
Use tempDir instead of liberty.temp.dir for creating test directory.

Add `page_story_id` field to Photo type
Skip AttachContainerCmdExecTest on non-Linux systems


Fix redirect URI to include query string

Fix: Add space between messages in TimeScheduler3 error logs

Fix NPE when accessing declarations in QueryResults
Refactor: Simplify PatternTestTools comparison logic for exceptions.
WARN: Handle potential deadlock during zip export close (SHRINKWRAP-120)
Throw ProgramInvocationException if no jobs are executed in interactive mode.
Remove Incubating annotation from StackdriverMeterRegistry
Use ROOT_PATH for select calls in Contents view.
Improve JEEContext performance by caching request content
Improve ResourceResolver with relative path resolution
Improve error handling and type checking in RestRouter.
Simplify IINC instruction handling
Elastic Load Balancer: Health checks don't require listeners.
Use static method for RedisDialect.toBytes() call
Increase StringBuilder capacity in DefaultChannelId.newLongValue() method

Use local host address for REST API tests
Use `getTokenizingCharacters()` instead of hardcoded string for tokenization characters
Exposes converters module and provides GzipFileConverter implementation
Fix buffer size calculation in LZ4BlockInputStream

Improve LiteBlockingWaitStrategy by adding a check for signalNeeded
Resize back and front buffers and trigger full redraw after resizing
Handle ClosedChannelException instead of AsynchronousCloseException in NioWorker.
Fix: Temporarily sleep in ClassLoaderAdapterCallback to avoid race condition during type proxying.

Fix tokenization of PDF dictionaries.
Add isInputMethodTarget() method to AbstractTextViewAssert
Use final for wrapper and dropwizardModule fields

Refactor: Remove unnecessary exception handling in commandsTest method
Fix delegate task handling in MVELLifeCycleManager.
Fix parsing of VLCJ info lines with potential extra ':' characters
Use Locale.ENGLISH for area finder case conversion
Improve stream closure handling by adding a timeout
Format elapsed time in milliseconds instead of nanoseconds


Add @Ignore annotation to EdgeTest
Fix comparison of SPARQL LANG predicates with XSD string values.\n\nThis commit addresses a bug where comparisons involving SPARQL LANG predicates and XSD strings were not handled correctly. It introduces a temporary hotfix that returns the original term in these specific cases.

Add UUID generation for PushApplications
Allow customization of CSV separator and quote character

Remove unused `registerSubsystemDeploymentWriter` method.
Add support for cookie expiry date
Add support for UNION type in StreamReaders

Chore: Disable default creation of external Ingress/Routes for LoadBalancer services
Initialize IOSServer with configuration options
Use existing order ID for trade creation if available
Add getter for TransactionController

Remove redundant scope groupId calls in CommerceAccountAddressAdminDisplayContext
Add ensuring clean session capability for Internet Explorer.\n\nThis adds a new capability to ensure a clean session when using the Internet Explorer driver.
Update LoggingAfterJobListener to use JOBLOG_FILENAME instead of JOBNAME in MDC.

Refactor: RichFacesTest - Disable unnecessary Ajax checks in repeatUntil method
Use CommonContext.IPC_CHANNEL for channel in DriverConductor.getOrCreateReceiveChannelEndpoint and SubscriberPos allocation.
Improve efficiency of ComponentCompaction#isBlank()
Remove unused @Resource annotation
Add support for registering event listeners
Use 'getId()' instead of 'getDownloadableId()' for consistency.



Refactor: Use enum values for StaxEvent array
Add javadoc for getTemporaryPath() method.\n\nAlso update since tag.
Add fast QR Code configuration option for optimized speed
Only locate additional codebases if any were discovered
Refine trimming of annotation times in KWICPanel
Update dependency injection for ProcessArchiveService

Add exception for robot death by old age
Fix compass sensor detection for Samsung devices

Refactor: Simplify SHELL_ERRORS flag handling in ShellErrorHandler
Allow simultaneous duplicates in queries (#123)

Improve ClusterEventDissector output by adding endpoint information

Add methods for handling flyweight tasks in QueueListener.
Ensure SimpleConfigList constructor uses correct ResolveStatus
Add @SlowTest annotation to testExecutionShouldNotTimeoutIfTraceable_WithMultiCallback


Fix: Initialize hashCode with 1 instead of 0.

Handle wrong content-type with optional failOnError option
Reduce number of paths and print hedge error distribution

Refactor: Replace direct Main method call with instantiated object and handleCommandLine
Update shard URLs to use Garena paths

Add debug logging for loadMetadata calls
Improve error handling in ServletContainerInitializerRegistrationBean
Simplify constructor of TupleClassMeta
Update version number to 3.1
Fix: Set saturated flag before checking reflexivity in `ReflexivityCheckingTest`.

Refactor: Check for and remove duplicate conditions in LogicTransformer
Add support for version property and versioning functionality. (Placeholder implementation)
Ensure batch start is only triggered when there are events for processing

Ignore open stream analysis in main methods
Increase queue size for audit log entries from 100 to 10000.
Ensure I18nLanguage comparison includes external flag
Fix testNonExistentHost to use invalid hostname instead of unknown.host.
Handle potential null error stream in SegmentHTTPApi
Refactor: Remove unnecessary public keyword from OnExpressionCondition
Improve error handling in Operators.java by removing unnecessary bubble calls and logging dropped errors at ERROR level.
Fix: Ensure display name is set correctly in Description constructor

Add support for multiple `type` parameters in NearbySearchRequest
Close selector and server properly on shutdown.
ODocument: Ensure fields are passed in pairs
Execute delete operation without access check for AccessSetUpdate

Store parsed server version number for efficiency
Improve NodeCmd output clarity by separating Gossip and Generation No when gossip is not initialized.
Refine shader path display in property handler
Fix: Use beacon object instead of concatenated MAC address for connection
Fix header height calculation for all-day events
Add toString() method to PostmarkResponse for easier debugging.
Prevent re-association of HTTP contexts with bundles if `allowReAsssociation` is false.
Improve HetatomImpl search for atom by name
Handle IOException in SSL downgrade path (#issue_number)

Improve profiler logging with configuration details
Improve MapFlatSelectiveStreamReader by building value block before returning
Deprecate `doapply` and `apply` methods in DBCollection\n\nThese methods are deprecated as they will be removed in a future release. There is no replacement for these functionalities.
Add sorting to JobInstance fetch requests

Refactor: Rename addItem() to reAddItem() and add warning comment about potential exceptions
Add checkClosed() before accessing cache entries
Improve OperationRequestCompleter by returning buffer length in specific scenario.

Refactor: Remove unnecessary last voted for reset in FollowerState.


Fix whitespace in RollbackTest doc comment
Add refresh method test and update AllTypes class.
Handle KafkaSupervisor notices with Throwable instead of Exception
Fix TFGraph import issues for specific Zoo Models\n\nAddresses issues with `xlnet_cased_L-24_H-1024_A-16` and `cifar10_gan_85` models.
Clarify TargetType enum descriptions

Update JDBC driver version to 1.3.5

Improve PushManager shutdown by returning all unsent notifications.

Add option to disable JPA annotations generation

Add support for PUT requests
Safely rollback transactions in ObjectifyImpl.
Improve URLPackageScanner by correctly extracting class names from file paths.
Use a single CountDownLatch for job execution synchronization in JobManagerTest.

Add empty lines for better readability.
Add logging for skipped pattern matching in NewJavaFileWriter
Disable new bytecode generator.

Add support for closing additional closeables in Lifecycle.trying
Refactor: Improve handling of metric selection in JPerfmonParamsDialog
Improve documentation for PdfBoxGraphics2D.create()

Fix: Language label font size in SpellAndGrammarCheckDialog
Add test for path creation ignoring empty strings



Simplify Emitter thread interruption handling
Add support for observable and interest handling
Remove deprecated code from getScrollListWindowHeight method.
Use Deque instead of Stack for ImportStack
Clarify disableDefaults parameter description

Add method for creating temporary files
Remove SQLException from insert and delete methods
Use section symbol in pattern and string representations.
Handle role loading errors with InternalServerErrorException
Increase maximum processing time for job service tests
Add onload attribute transformation for archival URLs
Sort subunit stoichiometries in descending order and update formula string accordingly.

Remove JAVA_OPTS from environment when launching Spring Boot CLI applications
Fix potential index underflow in RoundRobinExecuteWithFailover
Add JsonCreator for GrokPattern

Support loopback address via -b argument without XML changes (#WFLY-248)
Add more informative error message for NodeUtil#getExpressionFromPropertyAccess.
Use Interceptor priority for CustomPathResolver
Use internal network by default for Termd Build Driver (configurable)
Add @SuppressWarnings("unchecked") to fromJSON method

Handle missing topics in KafkaSupervisor.
Only inject LiveInjector if Eclipse needs patching


Improve UFS attribute retrieval in WebInterfaceGeneralServlet\n\nThis commit refactors the way UFS attributes are retrieved for display in the web interface.\n\n- It now attempts to create a root UFS instance instead of relying on a specific mount table root.\n- Error handling is added to gracefully handle failures during UFS creation.\n- The `setUfsAttributes` method is introduced to encapsulate the UFS retrieval logic.
Fix: Use correct element locator in CorrectEventFiringTest
Remove unused logger instance from IndexedSet class.
Refactor: Simplify error handling in Finder class

Load class method now returns null on failure

Remove unused postEffectShaderManager parameter from cullAndRender
Use ContextFactory to create scripting context
Add mapping for JDBC type DECIMAL (#123)
Remove unused stopUFS method in LocalTachyonCluster



Add marker details and title to map
Fix: Correct boolean parameters order for batchSynonym API call


Introduce NOT_A_CHAR constant for handling invalid Unicode values in CSV.
Renamed TAG_H_POSITIONING_ERROR to TAG_H_POSITIONING_ERROR for clarity.
Deprecate `ZMQ.poller()` and `ZMQ.poller(int)` methods. Use ZContext or Context.poller instead.
Add test for spaces in LDAP URL
Ensure timestamp sorting in TestFilters.
Fix: GroupProperty handling of null values
Refactor: Use andThenConsume instead of consumeAfter and atLastConsume

Remove redundant test case for cache management with two databases
Remove unused Description class from MetricOptions.
Use SEGMENT_FILE_LENGTH_DEFAULT for LOW_STORAGE_SPACE_THRESHOLD_DEFAULT
Fix potential NPE in AsyncTcpSocketImpl.java
Update default servlet version to 2.5

Make all domain properties optional in GrailsDomainBinder.
Use configured locale for gallery info retrieval
Add missing characters to valid word list pattern
Add default group ID for fraction list.
Increase minimum allowed single upload size to 5MB
Flush write context after updating metadata in OffHeapValueHolder
Fix ProductResultParser to handle UPC-E correctly.

Add Envjs wait for browser execution
Add isEnabled(), setEnabled(), isRegisterManagement() and deprecated shouldRegisterManagement(). (#<issue number>)
Fix: Use 't.header' instead of 't.header_' in MessageSerializer serialization.
Refactor ST_Explode: Find first geometry column instead of SRID

Improve BlockLockManager by adding initialization of ClientRWLocks and updating LockRecord mapping.
Read the remaining of the current block in FileInStream
Only remove artifacts without connected build records from cleanup.
Add shutdown logic for executor in FileCacheStore.
Copy required properties for service chaining in TargetHandler
Fix: Use build start date instead of string representation for buildStarted in ExtractorUtils.
Remove unnecessary TODO comment and improve dependency management for ImagePicker generator.

Use roleLocalService instead of rolePersistence for fetching roles
Remove unnecessary logging level adjustments in DefaultSocializeProvider.



Add support for setting title color
Add check for empty filter list in handleScanResult
Fix regex pattern compilation for case-insensitive matching\n\nThe current implementation uses (?u) which breaks Dutch rules.  This commit changes it to a workaround until a new release with a fix is available.
Refactor: Remove unnecessary public keyword from PerLineTailSplitter class
Add sleep after starting MindmapsEngineServer

Handle internal errors during authentication gracefully




Make DefaultIndexRedirect a regular class instead of abstract
Ensure OAuth parameters are included in multipart requests for correct signing.
Remove unnecessary System.out.println statement in SomeSteps constructor

Fix user creation URL
Rename setUp method to setUpChrome for clarity
Cleanup Speak Service reference
Make runtimeBrowserString and setTestContext protected
Remove unused import for Sets
Use container's default ReleaseId if pom properties are not found.
Use final keyword for lock object
Fix: Ensure correct comparison order in SingularityTaskMetadata

Add error handling for metric expression evaluation.
Refactor: Initialize size, onColor and offColor in constructor

Notify on preference header selection
Remove unnecessary RuntimeException throws in Fmrc.java
Improve documentation for CacheCreateConfigOperation regarding createAlsoOnOthers behavior.
Improve efficiency of table clearing by using a direct connection
Deprecate closeQuietly() due to misuse and potential exception swallowing

Implement faster atan2 approximation using a custom algorithm.\n\nThis replaces the standard Math.atan2 with a faster implementation based on a formula from StackExchange.\nThe new implementation can be enabled by setting the Options.FASTMATH flag.
Return HDFSDataset.Builder instead of creating it directly in newHDFSDataset()
Improve timer execution logic by checking for persistent timer executions
Use StringBuilder instead of StringBuffer for better performance.
Reorder nodes in query traversal for improved efficiency.
Refactor: Prefer exact method matches over similar ones in ReflectTest
Fix: Idempotent close for JavacProcessingEnvironment and workaround for compiler bug
Refactor: Standardize flag names in DistributorFlags config.

Fix: Use package name for class access in ClassBundleService call.
Improve responsiveness of ScanProgressDialog by using SwingUtilities.invokeLater for showing progress.
Add overloaded loop() methods for gain and position control
Remove unused 'data' resource
Add missing license header and author comment to PlanConfiguration.
Format thread dump output with quotes around thread names
Improve error handling in ReflectionUtil#getDomainClass()
Refactor: Cleanup logic in Whirlpool

Fix bug in LSMSolver where no L1 penalty or bounds led to incorrect diagonal addition.
Add retry logic for HTTP client exceptions except 401 Unauthorized
Remove unused LrsObject class

Use mkdirs() instead of mkdir() to create server temporary directory.
Add Jackson annotations for ErrorMessage serialization.

Introduce `controlWindow` and `currentProduct` as nullable fields
Disable debug mode in Shippo API client
Add isSupervised method and update classification logic
Fix screenshot size comparison in NewSessionTest
Add sudo method for simplified consumer execution under admin.
Make credential reference secret volatile


Use OpenCms resource manager to check for container pages in CmsSolrIndex.
Remove unnecessary dependencies from WarTestProbeBuilderImpl
Fix heartbeat frame order in DriverPublication

Ensure termLength is cast to long before multiplication
Improve error messages in FileSystem class.
Increase cache clean time and add @Ignore for testCacheExpiration due to concurrency issues.
Ensure CachingRemoteWebElement has correct parent and file detector
Handle missing methods gracefully in ConstantsWithLookupProxy.
Use LinkedHashMap for FlatProperties to preserve property order
JDBC_PING: Improve logging for initialization SQL failure
Use safe() for package name in scanClasses()
Use AlwaysIncreasingSystem2 for DbTester in CreateActionTest
Fix potential ArrayIndexOutOfBoundsException in Utils.getBytes()
Deprecate `isSubscibe()` method and add TODO for removal
Add 'Utils' and 'Records' to reserved type names in JavaCodeGen.
Log marshalled payload string only if trace level is enabled

Ensure every function has at least one output parameter.
Improve FlowRunnerTest2 by using a separate thread and cleaning up the runner instance.
Refine error handling in JSpecRunner (catch AssertionErrors and add more context)


Only add completed limit orders to open orders list.
Fix: Call superclass toString in generated toString method

Add functionality for repeating last command with Enter key
Fix ServerEventTest.java: Use correct InetAddress for testing.
Refactor: Flatten paragraph blocks in rendering
Fix: Handle missing classList resource file in CoverageTest.
Update SessionSummary state mapping for COMMITTED state
Improve resampling precision handling


Start monitoring CPU and memory usage with sensors.
Enable/disable 'Scan headers in all requests' based on 'Scan HTTP Headers' selection.
Use ConcurrentHashMap for storing executor services

Update default service name to 'mySample'
Deprecate HttpServletRequest and HttpServletResponse access in Context interface. (#... )\n\nThese methods are deprecated because they directly refer to the servlet API which is not always necessary or desirable. Users should rely on the provided context objects instead.\n\nThis change will eventually remove these methods completely.
Improve clarity in user prompts for file conversion count
Resolve app name before parsing parameters and launching activity
Refine LeaderAppointer initialization based on start action
Improve SharedHealthCheckRegistries.getDefault() method by returning null when default registry is unset.
Use `time` instead of `jdate` in JDateTime hashCode calculation
Improve pattern matching for Nullable and Nonnull annotations

Rename `toStringEquals` method to `strEquals` for better clarity.

Add STUN support and related tests.
Add proxy configuration to NewSessionPayload
Refactor: Use Throwables for exception propagation in DefaultLogicResult

Add test for assertArrayTypeWithNullIsNullSafe
Only run rename view test on single-user server role (>= v3.4)
Handle Ctrl+C interrupt gracefully in CliTableauResultView
Add isRuntime method to ThreadPoolResourceDefinition
Use Google Volt Patches Stopwatch instead of Google Common Stopwatch
Handle script element parse errors and increment counter
Clarify purpose of AndroidTestFinder
Add ResourceAccessor and initialize it in execute method
Simplify ResourceConfigApplier method signature

Fix: Ensure Postconditions.checkNullReference always throws an exception.


Change log level to debug from INFO in LoggingTest

Fix: Correctly update active transport count in PublicationImage.\n\nPreviously, the active transport count was not always correctly updated. This commit fixes the issue by ensuring that the active transport count is only incremented if the last frame time is within the image liveness timeout. Additionally, the commit updates the way the active transport count is written to the log buffer descriptor for better accuracy.
Fix: Ensure proper room leaving even if sendStanza() fails
Fix bug in FastAdapter.java where loop for adding items was not starting at the correct position

Use HTTPS for S3 object URLs
Use Arrays.hashCode for RawType hashCode calculation
Fix: Use onClose field instead of calling it twice




Handle potential UTF-8 surrogate issues in StringSerializer
Refactor: Handle IllegalStateException more gracefully in DatastoreImpl
Use Number instead of Double for count value in CountCommandResult.
Warn user about deprecated project file compilation syntax.

Remove unused ScriptObjectMirror dependency

Add flush to Elasticsearch after importing VCF data. (#123)
Disable disk threshold allocation in Elasticsearch container for tests.


Remove unused PropertiesEscapeLevel enum
Add waitAllForSafeState after adding a new instance
Use time as nonce by default for synchronization and handle nonces within same time interval.
Fix: Use correct class loader when generating extended parser classes\n\nThis commit fixes an issue where the generated extended parser classes were using the wrong class loader.\n\nInstead of relying on the default class loader, the `ClassWriter` now uses the class loader from the parent class of the transformed parser class. This ensures that the generated classes can correctly access resources and dependencies defined in the parent class.
Prevent repeated shutdown requests and improve shutdown process.
Fix: Check if connection is open before closing.
Make SurfaceLayerGL's surface field final
Remove unnecessary time logging from FramerManager.
Handle null response header case in RestClient#getHeaders()
Fix ECKey ASN.1 parsing to handle potential type mismatches

Refactor: Evaluate property invariants during lambda execution



Refactor: Remove unnecessary whitespace trimming logic


Fix: Close writer only when output is used.

Improve PlaceRegistry logging and use ploid instead of getOid in unmapPlaceManager()
Use Chronology from DateTime for range calculation in Period.

Change ServerRemoveHandler to run at RUNTIME stage
Fix: Ensure all shares have equal length



Improve LegacyAddress constructor documentation


Fix: Handle null LocalVariableInfo in InstructionSet.isDeclaration()
Add toString method to BootChargingStationCommand
Refactor: Handle FileDatastore check in isFileFound method.
Refactor HelpFlags to use forEach instead of manual iteration
Add getToolStrip() method to FeatureAttributeWindow
Run pre-crawling plugins before initial URL navigation.

Add methods to retrieve first header and its value
Use PacketManager.writePkg instead of PacketManager.write for sending packets.

Fix: Handle null titleBarBackground in setBackground()
Improve error handling in NatRegMrCall by providing more context on memory registration failure.
Renamed `activity_type` property to `type`
Make CampAST.emit() final and return toString() for now.
Fix ORDER BY optimization with replicated tables in outer joins
Add sequence number field to CommonLogFields
Allow URLs starting with 'https:' in BoxBrowser.
Make CanvasTestBase abstract
Add curly braces around interface definition for consistency
Fix: Ensure case-insensitive comparison for attributes using Locale.
Fix typo in DisposableHelper.dispose() method comment
Fix: Clarify input/output direction in Streams methods.

Use static hook for onCreate
Fix: Address potential variable importance calculation issue due to data shuffling
Refactor FindAllQuery to handle result set iteration more efficiently and ensure expected row count.
Improve class loading by using loadClass() for findClass()
Improve logging for IconEnricher
Add -jmx option to LargeState test.
Clarify Android SDK platform version parameter description
Fix: Change error code for StaleElementReferenceException from 13 to 10.
Refactor: Use checkIfClassIsModel and isClassModel instead of direct OpenEngSBModel checks
Fix calculation of cookie expiration time for maxAge
Make JAXBUtil a singleton
Refine request counting logic in BrowserMobProxyTest
Fix TextFieldSkin layout and binding issues

Rename TransactionRuleBuilder from AuthorityRuleBuilder
Improved loading of activejdbc.properties by searching multiple resources.
Update DiameterStackMultiplexerAS7 interface with @Local annotation.

Add phenotype validator and individual creation retries option

Refactor KafkaClient: Refresh configuration on server or groupId modifications
LazyPatternBy: Implement toString and synchronize nested retrieval.
Update app logo URL to use href attribute.
Fix: Use correct version in QueryRequest.


Fix: Return empty list for properties in ListItems
Refactor: Change how `MutableBySettingFieldOfField` sets indirect fields.
Update javax.* imports to jakarta.*

Use system property 'm2.repo' for Maven repository directory


Add since tag for ModelObjectWithChildren interface (#... )
Refactor: Remove unnecessary delay generation check in activateGap
Fix: Use parseUnsignedLong for channel ID matching in EntityBuilder
Fix potential NPE in generating foreign key references in changelog
Add validation for surrogate pairs in DatabaseUtils#escapeStringSql
Fix: Don't initialize RippleView in edit mode
Refactor: Simplify getFileContent method
Pass TSMeta through tree processing after storage.
Fix: Use RepositoryService instead of RuntimeService for creating deployments
Add test for none rows with other threads and exact queries
Refactor SqlgStartStepBarrier to use Traverser.Admin and remove deprecated code.
Fix List constructor logic for handling element tags
Simplify message handling completion logic in Puller
Improve formatting of low storage warning message.
Fix: Use click() instead of setSelected() for radio buttons
Update GRADLE_VERSIONS list with Gradle 6.2
Add SPARQL service description namespace and update properties accordingly
Use long literals for calculation in getMaxMemory




Refactor: Use 'disabledTransforms' instead of 'blacklist' for global AST transformations.
Reduce thread pool size in TestMultiThread for performance improvement.
Handle remote link close and detach events
Disable testCreateBucketedTable due to timeout issues.
Improve bitmap decoding concurrency with a global lock.
Rename parameter '_predicate' to '_a' for consistency.
Fix: TransitionSet clone now properly sets transition durations.
Fix handling of null columns in Index toString()
Log event type and body when handler is null
Remove @Ignore from RSS and Atom tests (JENKINS-59167)
Fix SoundManager drain() issue causing thread hangs and delays.
Fix: Throw more informative exception when creating empty JavaFileObjects
Fix: LocalDate.ofYearDay test range
Refactor: Update UserProfile constructor parameters
Refactor: Simplify loading of EvolutionResult
Ignore StageTest due to pending refactoring.
Remove ParallelTest category from WanRESTTest
Change flags field type from short to byte
Update OptimalCECPParameters with generics
Fix rendering mode serialization
Synchronize `myLooper()` method in ShadowLooper
Add separator and filename to AbstractFile toString()
Refactor: Only append HTML if it's not empty.

Fix: Use forward slashes in file paths

Add support for describing SelenideElements as well as WebElements.\n\nThis change allows Describe to work correctly with both types of elements.\n
Add setData method to SessionInfo class to handle rules bytes.
Use CPU time instead of scheduled time for plan statistics calculation.
Make CloudSpannerOAuthUtil non-instantiable

Add methods to get URI and contents as stream for File System Support

Improve saved output file naming in ExtendedEmailPublisher.
Refactor: Move WeaselProgram out of engine package
Fix: Correctly extract missing resource names
Use constant-time comparison for salt and hash in HashData equals method
Refactor DocumentGraphMapper: Handle node mapping based on `annis::node_type` attribute.
Use JCommander's parse method instead of creating an unused object.
Improve type safety in ProxyDefaultMethodHandle.invokeWithArguments()

Add check for type parameter in method calls
Fix reading class file header on JDK7

Improve DagManager's hasRunningJobs method for clarity and accuracy
Remove shutdown hook for graph database
Use string instead of int for LinkedIn user ID in Identity creation.
WorkerOptions: Set default workers amount to 1
Fix typo in formatter documentation
Add security restriction lifting functionality to CLI

Fix CulpritsRecipientProvider for non-AbstractBuild runs
Fix AudioDisconnectEvent firing when not connected and close audio connection on disconnect
Use flipped buffer for data deserialization in InstallRequest
Add Preconditions check in JoinMatcher
Fix: Adjust content extraction range in SourceCode


Fix: Correct LakeBTC API URL
Add commit() method call to FileDownloadUtil.create()
Improve RefreshListener logic for event targeting (#1234)

Fix: Calculate endColumn correctly when there are errors on multiple lines.
Refactor: Use dedicated method for id element check in PairBasedRestfulRouter
Refine private identifier detection to exclude internal symbols ending in '__'
Make WebSocketSubscriptionTransport constructor public
Always return null for ViewProperty instances
Change error message in FileList activity to 'cancel' instead of 'error'

Change log level for interrupted sleep in OCommandExecutorScript from error to debug.
Add constructor with Jackson annotations for HostIdentifier


Improved parsing of GPSD data by attempting double parsing for non-time fields.
Simplify and minimise automata during normalisation
Add default artifact type for SVN connector configuration
Use TreeSet for storing command names in getAllCommandNames method for sorted output.
Use UNDEFINED instead of null for undefined type variables
Add CacheResultUnavailable exception and deprecate CacheResultUnavilable

Add parent and allowOverwrite flag in readItem function
Refactor: Change log level from DEBUG to TRACE for discovery entry related messages.
Use SpecificationsMapper for counting entities
Remove clear() method from StandardTitanGraph.
Use channelUri directly in ingress publication creation
Prevent modifying index names during column family definition updates
Add Javadoc comment to testCovtypeGBM method
Add name property to GeoJsonObject
Introduce conductor cycle threshold configuration

Add mgmt-users.properties file check in AddPropertiesUser
Remove task registration from Task.run() method.
Fix: Work around compiler bug in closing processor classloader
Handle GitHub display name retrieval
Add readForRecycle method to ByteBuf

Fix: Propagate error when it's not on subscribe and assert single failed call.
Update repackaging logic to use ArtifactHandler extension


Add constructor for Meta without assigned id
Improve sigmoid function and handle potential float precision issues in ObjFunction.

Fix spelling correction for "unswar" and add support for variants like "un[sz]war"
Refactor: Replace 'is' with 'in' for DataInput parameter
Refactor: Use single VelocityEngine instance in DevelopmentPlugin.
Improve thread usage in PlannerBenchmarkConfig

Start stats agent after construction and remove redundant OpsAgent registration
Refactor: Use PathTools.getPathsOfLengthUpto for more efficient path generation in Fingerprinter.

Handle file syncing for both zip and non-zip files
Refactor: Rename `getAllExtensions` to `getExtensions` in UserExtensionsTest

Allow specifying a single entity in HibernateBundle constructor.
Add handling for x-amz-acl header in PutObject requests
Refactor: Simplify scroll event handling in MapController
Add scope management for XVariableDeclaration initialization expression.
Improve handling of `ForAll` loops in VerificationTransformer.
Register plugins after initializing webview
Add version and lock time information to Transaction toString()
Remove debug logging from showMenuItem loop.
Remove debug logging from ExceptionProxy.constructExceptionForClass()

Warm up partitions in SmallClusterTest
Allow `singleton` option to accept boolean value
Use getPreviousSuccessfulBuild() instead of getPreviousNotFailedBuild() for dependency changes.
Remove reactor strategy configuration
Use scheduleWithFixedDelay instead of scheduleAtFixedRate for latency marking timer in StreamSource.
Reduce logging verbosity in SelectorOptimizer


Add node role information to disk usage stats

Add `--provision` flag when querying running Vagrant state
Prevent emitting notifications in ReadOnlyContext

Improve error handling in AbstractRemoteCommandObject by including response entity in exception message.
Fix: Ensure brand files are not empty before processing
Fix potential NPE when closing PythonPlanStreamer server
Remove unused usage count tracking from ClassDescriptor.
Add toString() method to FluxMaterialize
Add initialization logic and exception handling for AutoReloadingConfiguration.
Log exception when creating AgiScript instances.
Update author name in ProductRegisterEndPoint.java
Simplify PUBREC constructor by removing default constructor.
Add test for large date format in PactDslJsonBody
Remove unnecessary System.err.println
Use Msc.isStandalone() to determine production mode.



Reduce `NUM_FILES_PER_CHUNK` for improved load leveling and reduce ZipFile overhead (#123)

Allow testNotifyArgument for protocol version 2 and server versions before 9.0
Fix: Remove expired listeners from rejected notification listener list
Add logging for FeatureRequirement assumptions
Enable StrictMode for debugging

Add toString method and handle empty location
Clarify DiskLruCache documentation regarding editing entries
Update thread pool size in AsyncScalr
Refactor: Use generic type parameter for Consumer in afterNodeLoaded method
Fix incomplete chunks serialization for CADD scores
Add support for filtering Java source files during buildAll
Cleanup attachments after sub-deployment stop.
Fix: Return `changed` instead of `removed` in `removeAll` method
Improve JSON Schema Validation Test by separating repositories with different schemas
Fix LazilyLoadedCtor to handle cases where class constructor is null.
Update candidate term ID in cluster mark file after election.
Only create analysis summary if findbugs.noSummary is not set
Remove userId field from OrganizationDto
Refactor: Wait for Opera idle or window load based on URL change.
Refactor: Improve clarity and consistency in toDate method
Simplify Script execution handling.
Restrict Partially Constructed Object Access detection to non-private methods
Update copyright year

Improve image progress reporting in IFFImageWriter

Increase open files limit to 1024
Improve backpressure handling by adding a queue and processing pending messages when channel becomes writable.
Refactor: Remove redundant type validation for ShardingProperties
Fix: Use Context instead of Fragment in SettingService creation.

Refactor: Remove unnecessary flush from rename operation.
Add access check cache to Permit.Fake class
Fix potential buffer overflow in writeTruncatedEntry by accounting for OBJECT_HEADER_BYTES.
Improve cancellation handling in SqlStageExecution
Remove debug print statement for CfDef object.

Change TreeNodeLongClickListener to return boolean.
Fix: Ensure ValidationFailed has at least one problem
Improve temp file handling in ServerIntegrationTestBase
Remove unused code for iterables and streams
Add PriorityResource binding
Fix: Correctly update old Response object in FunctionContext's doFilter method.

Increase benchmark slice size for better performance


Add support for specifying a static constructor name in @Data annotation

Only add tools.jar to classloader when compiling
Only log sending messages if debug level is enabled
Add compareAndSet method for atomic sequence updates



Update SearchUI to refresh user information on page load.
Fix: Adjust initial store index for method receiver.



Refactor: Improve Timeslot Duration Validation in SseBuilder
Increase timeout for NettyTransportTest methods
Fix: Access token check for empty string
Use String for attribute names in CmdLineAnnotatorUtils
Improve Task.isRunning() logic
Fix: Ensure cluster formation is always awaited

Fix: Ensure getPerformanceNumbers returns 'not available' for missing or empty cases.
Refactor: Separate layer state from shared state


Add X-Text-Delivered header to LogResource

Use native method for interrupted() call
Improve RollingUpdateOpFactory error messages for PORT_CONFLICT.

Deprecate all Map interface methods as ContentMap is immutable.
Fix: Prevent newline and tab characters in normalized URIs

