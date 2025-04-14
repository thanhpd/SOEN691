Add categorization of errors in ErrorList
Make logger field protected
Log configuration at debug level instead of info
Add helper methods for getting master and worker addresses from configuration
Simplify random number generation in getDbConfigUrl()
Improve PRIMARY_KEY_CONSTRAINT_VIOLATION_MESSAGE pattern for Fongo compatibility
Add initial URL navigation before element extraction tests
Fix logging of frame size in ReceiverUdpChannelTransport
Refactor: Use add() instead of set() in fromStringList method
FLINK-23139: Handle null stateRegistry in discardState() method

Use thread-local SecurityContextHolder for testing context
Add constructor for JedisShardInfo with host, name, port, timeout and weight
Make loadInitialData public

Add load/unload/reload functionality to MaterialDataPager



Refactor: Simplify OHttpResponse content type handling
Add captions to search dialog elements

Improve readability of comparison output in CgIsaParamTest.
Use Hamcrest MatcherAssert for assertions

Remove BIAS_FEATURE from SentFeatureExtractor
Update management version numbers to 3.0.0
Skip nested CityObjects during unmarshalling
Use canonical hostname for local worker address resolution
Improve ClientConnectionsShutdown by removing unnecessary promise and using channel.closeFuture() instead.

Use '|' instead of '|' for channel separator in replay URI
Add verbose logging option
Log level change for VirtualType property removal
Use Class<> for type parameter in mocking.

Simplify type extraction for EnclosedExpr nodes
Ensure extensions are terminated before shutting down threads
Add examples of array initialization with different types and syntaxes.
Allow Postgres container to participate in XA transactions by default.
Fix typo in test suite name
Fix date formatting in DurationBusinessCalendarTest
Fix handling of empty reference and alternate alleles in Variant constructor.

Remove unused SpinnerAdapter import
Fix escaping of sequence names for PostgreSQL\n\nPreviously, all database types were stripping quotes from sequence names when using the next value function. This caused issues with PostgreSQL as it relies on quotes around identifiers containing mixed-case characters.\n\nThis commit now correctly handles PostgreSQL by leaving quotes in place while stripping them for other database types.
Synchronize STIXSchema validation for thread safety.
Add support for Byte type in ColumnValidationReader
Use UUID for unique lock holder identity
Fix: Scrap view type handling
Fix: Remove unstable test case for Cyrillic name in Chrome
Use ApplicationClassLoader's class loader for loading system classes
Improve documentation for XMPPConnection#addSyncPacketListener and XMPPConnection#addAsyncPacketListener.
Clarify Layer.setRotation() documentation
Fix query result count when using limit option
Fix escaping of source file name in generated code
Add `started()` method to check if REPL is running
Add tests for keyboard behavior in StyledTextAreaBehaviorTest

Fix: Clear current span before starting a new trace in ZipkinSpanListener
Reduce test data fraction in DeepLearningProstateTest
Make ClientFactory public
Fix: Calculate target frames based on desired frame rate (60fps)
Fix: Ensure correct byte array conversion for attachments
Use canRead() instead of isFile() to check for tools jar readability.
Exclude 'nobs' metric from ModelMetrics output.
Improve GetSchemaInfoCommand by handling both Node and Relationship entity types.
Update $Date
Fix tick handling in ControllableRunner
Ensure client is closed after executing goals.
Set Tomcat base directory to target/tomcat-{test class name}
Fix TestIgnorance logic for handling service versions
Refactor test cases for string concatenation and variable manipulation.
Fix ObjectUnderFileSystem getObjectListingChunk logic for folders
Allow visiting fragment spreads in VariableTypesMatchRule
Increase buffer size for server choice read
Refactor: Simplify getMaxInitialId method using uniqueResult()


Disable Spark UI and run tests locally
Add delay before stopping Instruments for RealDevice sessions
Update JavaFormatterTest with new test data
Improve documentation for IClassWriter interface
Refactor: Simplify appendOp logic for better readability.
Update DruidMaster log messages for better clarity
Improve Percentile series spec with literal representation
Refactor: Remove deprecated Expect.apply() implementation
Remove unnecessary @Slf4j annotation from SiteAwareBuilder
Re-add EnglishWordRepeatRule for English language support.
Fix type handling for boolean values in Proto class.
Fix: Use proper closing tag for gwt:property meta tag
Use source info from method node for assignments in class rewrite
Disable sound for Upload Service notification channel if ringtone is disabled
Handle unexpected exceptions in getChoice() method.
Update protocol version to 5.1
Add methods for ending tags
Add check for Steam running status and update init method.
Fix testExceed128Connections to work with Renjin
Ensure ProcessInstance retrieval includes signals
Warn instead of debug when multiple kubeconfig files are found
Remove test for SMILES format
Add support for Conscrypt on Android platforms.
Update query handling and add corpus selection check in TigerQueryBuilderCanvas
Use 'this' keyword for accessing instance variables in AbstractPatternRule.

Remove unnecessary RapidoidInitializer.initialize() call
Remove unused MessageContent and InputStreamEntity imports
Add attrX method for creating AttributeExpression
Fix: DescribeCommandTagsIntegrationTest now correctly identifies lightweight tag
Remove redundant @Override annotation
Improve efficiency of key decryption by reading data in chunks matching the required size.
Fix: Ensure CacheManager instances are closed correctly based on classloader.
Handle potential NullPointerException during shutdownNow()
Use URI instead of toExternalForm for file resolution in RepositoryPersistenceTest
Refactor: Access aggregator and collector fragments through core object.

Add connectionId to MpScheduler message
Fix: Remove unnecessary empty line in Violation constructor
Add test for inserting Person with null key reference and select it.
Rename `getCachingAndUpdatingParserHolder()` to `getCachingAndUpdatingParser()` and mark it as deprecated.
Add tax ID information to CustomsDeclaration test data
Refactor: Update AccessLevel enum values for clarity and consistency
Introduce default implementation for IIndentationInformation
Disable anonymous access on Resource Server

Fix typo in glue path for postgresdb tests
Ensure directory creation and file writing succeed in TestDataContextUtils
Remove debug logging from DataSetTableIterator.

Update raster layer opacity handling
Fix DockerService.getNetworkGateway() to handle cases where docker-machine is not running
Make Connectivity.create() protected
Add check for null connectUri before starting acceptor loop
Ensure clientName is initialized after appender starts

Retry remote requests with exponential backoff and timeout handling.

Improve error message for mocking non-private/final classes
Refactor: Simplify ProcessInstanceIdCommand interface
Simplify invalid packet counter increment
Fix: Correct 'row' and 'column' references in Matrix insert/removeColumn methods
SONARPLUGINS-3097: Handle empty line at end of file in Git blame output

Improve LifecycleManagerStoreImpl by simplifying binding and tracking logic.
Use an explicit loop instead of relying on a for-each loop.
Use 'LIST' instead of 'string' for enum values in AllowableListValues
Add serialization check for MojoModel
Add support for distributed group execution
Use username and password for hostbased authentication
Refactor: Simplify history management in SwingTab
Make Encryption a final class.
Refactor: Use closePreparedStatement instead of closeBatch for PreparedStatements
Fix typo in `browseFrom` method and add deprecation for old spelling
Use Sequences.simple for SegmentMetadataQueryRunnerFactory output.
Use HashMap instead of Hashtable for properties
Fix: Correct grammar in warning dialog about remote database migrations.
Fix handling of categorical features with enum values outside training range.
Ignore testAnonymousType due to unreliability on integration server
Refactor: Prioritize user property for random seed over task attribute
Fix: Reset image name on band element if chart generation fails.
Make ThriftCollection case-insensitive for name lookup
Remove unnecessary @NotNull annotation
Add utility function for asserting floats with default tolerance
Ensure path starts with '/' when using @Path annotation
Add process time gauges to JHipsterMetricsEndpoint
Improve ResizeSettings validation with a delta value for float comparison.
Only schedule updates if a memory cache is set.
Fix: Allow non-positive header IDs (JIRA-1234)
Fix PackrReduce handling of directory inputs
Update master thrift shutdown timeout unit to seconds
Increase parameters for motion estimation in ExampleVideoMosaic.
Add note about missing max request size validation
Fix classpath string splitting on Windows
Refine log viewer display by removing date from simulation time

Refine logging levels for addon copying operations.
Add clearViewCached method to AbsViewHolder
Increase test duration for ArchiveRecordingLoadTest
Add CACHE_MODE_LOCAL flag to WriteCommand invocation context

Use Log.getStackTraceString() instead of throwable.toString() for better stack trace logging.
Add rule for PRP_RB_NO_VB and prioritize more specific rules.



Revert: Load function library on metadata reload
Fix TypeVariableCycleTest by changing annotation from TA to TB
Disable expression cache in eager mode
Fix NaN handling in Sum NodeList implementation
Add permissions for cluster config entries
Fix: Ensure ApnsConnectionListener is not null
Use ConcurrentHashMap for tracking finalizable objects.
Refine GaussianProposalDistribution for LinearRegressionTest

Use default watermark if previous committed states are not found.
Improve TimerTask calculation of next timeout by accounting for skipped intervals.
Add SourceVersion entries for Java 1.7 and 1.8

Avoid double processing of expressions in ExpressionAnalyzer
Make `CheckRange` more generic by using `Comparable<? super C>` instead of `Comparable<C>`. (#[issue number if applicable])
Update approvalsRequired in AsyncMultisubprocessTest
Update Metrics imports
Ensure database creation is executed with connection context.
Fix context classloader issue in WebXmlObserver
Handle null ResultPoint values when shifting points.
Use 'R' instead of 'H' for two-digit year format in JapaneseElementTest


Reduce assertion percentile for performance tests
Refine terminology expansion logic for zero count scenarios.
Fix issue #122: Ensure endDate is midnight for period counters
Fix: Handle chunk ID reversion in GameUpdateTask
Add more thorough testing for PoolTest with edge cases and potential issues.


Use getApplicationContext for ConnectivityManager in SdkUtils
Fix: Skip JPEG markers in RAF metadata reading
Fix typo in SpanBuilder startSpan documentation
Fix: Transient field handling in CRUDDocumentPhysicalTest\n\nThis commit addresses an issue where transient fields were not being properly handled during document reloading.\n\n- The test now explicitly reloads documents from both cache and disk.\n- Assertions verify that transient fields are present in the cache but absent when reloaded from disk, as expected according to schema definitions.
Make extractAsInputSource method final
Add getters and setters for ProxyFactoryConfig
Fix: Set result index for Tree nodes
Remove unused date format and simplify serializer creation


Improve TLS certificate hostname verification.
Update ReplaceIfSameOperation to store old value on success
Increase interactive job wait time from 5 seconds to 10 seconds.
Improve documentation for VMath.timesTranspose method and add equivalent alternative using a vector as matrix.
Fix termId calculation in ArchiveConductor
Fix: VerifyMojo - Prevent orphan check for Main.WebHome document
Prevent duplicate project creation during copy operation.
Refactor: Simplify setCoordinates method
Handle exceptions during log replication stop gracefully.
Enhance resource calendar view styling and behavior

Fix: Use eGet with resolve=false for batch linking references
Reduce ping interval for SimpleRoundRobinLBTest to ensure faster server updates.
Use an ArrayList instead of Collections.emptyList() for combined artifacts
Use tempDir instead of liberty.temp.dir for creating test directory.
Remove logging configuration from Server class
Add `page_story_id` field to Photo type

Add waiting and timeout functionality for tests
Update AccountHeader to reflect selected profile in selection list and fire onProfileChanged event.
Fix trailing slash redirection for RackFilter
Ensure registeredService and service are not null before bypass evaluation
Fix: Add space between task and error message in TimeScheduler3
Update connection event listeners for test configuration
Ensure QueryResults.getDeclarations() returns an empty map when no declarations exist.
Refactor: Simplify negation and inflected checks in PatternTestTools

Throw exception if program didn't contain Flink jobs and no environment was created.
Remove Incubating annotation from StackdriverMeterRegistry

Improve JEEContext performance by caching request content

Improve error handling and type checking in RestRouter


Use static method for RedisDialect.toBytes() instead of instance method
Increase StringBuilder capacity in DefaultChannelId.newLongValue() for better performance.
Add return statement for ATTRIBUTE_TYPE_GET_OWNERS_REQ case in TypeService.
Use local host address for RestHttpHelperTest

Exposes converters module and provides GzipFileConverter for FileConverter interface.
Fix buffer size calculation in LZ4BlockInputStream
Add getter for task speed in Listener4SpeedAssistExtend
Improve LiteBlockingWaitStrategy by adding a check for signalNeeded before entering the loop.
Resizes back and front buffers on resize and triggers full redraw.


Sort context values in CollectorContextTest
Fix whitespace handling in PRTokeniser.
Add isInputMethodTarget() method to AbstractTextViewAssert
Use final for wrapper and dropwizardModule fields
Remove transient keyword from Callback field
Refactor: Remove unnecessary exception handling in commandsTest

Fix parsing of VLCJ info lines with multiple colons
Use Locale.ENGLISH for area finder string conversion
Improve stream closure handling by adding a timeout
Format elapsed time in milliseconds instead of nanoseconds for ThreadLog.
Make Cycles constructor private
SEC: Use SecurityActions to get ClusterExecutor
Add @Ignore annotation to EdgeTest
Fix comparison of SPARQL LANG predicates with XSD strings.
Use acyclic joined graph for collateToSvg.
Add UUID generation and Set for application configurations
Allow customization of CSV separator and quote character
Use flatMapIterable instead of flatMap for reversing Flowable
Remove unused `registerSubsystemDeploymentWriter` method


Add suppression for REC_CATCH_EXCEPTION warning in ofBootLayer method.
Chore: Disable default creation of external Ingress/Routes for LoadBalancer services

Fix: Use existing order ID for trade creation
Add getter for TransactionController
Add documentation for WebSocketsExample
Remove redundant scopeGroupId calls in CommerceAddress retrieval
Add ensuring clean session capability for Internet Explorer
Use JOBLOG_FILENAME instead of JOBNAME for MDC logging
Use current instance of JadeTemplateEngine in extractors.
Refactor: RichFacesTest - Disable unnecessary AJAX checks during repeatUntil()
Use CommonContext.IPC_CHANNEL for channel in receive endpoint creation and subscription registration.
Improve efficiency of ComponentCompaction#isBlank() by using String methods instead of char arrays.
Remove unused @Resource annotation
Add event listener registration functionality

Fix: Read compressed data in chunks from InputStream
Add support for initializing FontIconTypefaceHolder with a File object.

Refactor: Use enum values for StaxEvent array

Add fast QR code detection configuration
Only locate additional codebases if any were discovered
Refine trimming of annotation times in KWICPanel

Handle null User-Agent string in ServiceThemeResolver
Add exception for robot death by old age
Fix compass sensor detection for Samsung devices
Use put() instead of add() for workQueue in TriggerTask
Refactor: Simplify SHELL_ERRORS handling
Allow simultaneous duplicates in queries (#123)
Refactor ThreadPoolEditor: Simplify keepAliveTimeout configuration
Improve ClusterEventDissector output by including endpoint information.
Add error handling when retrieving input type information.

Ensure SimpleConfigList constructor uses correct ResolveStatus
Add @SlowTest annotation to testExecutionShouldNotTimeoutIfTraceable_WithMultiCallback
Fix stale connection handling in PresentsClient
Allow specifying runtime root folder via system property
Fix: Initialize hashCode with a non-zero value

Handle wrong content type in AbstractSingleImageLayer with failOnError option
Reduce number of paths and print hedge error distribution statistics.

Refactor: Replace direct Main method call with app instance in FullTest
Update shard URLs to use Garena paths
Add method to retrieve received messages without waiting.
Add logging for loadMetadata calls with LoadMetadataOptions.
Improve error handling in ServletContainerInitializerRegistrationBean
Simplify constructor of TupleClassMeta
Update ArrayProxy version to 3.1
Fix: Set chain saturation before reflexivity check in ReflexivityCheckingTest
Improve performance by setting buffer position before writing in AsyncFile.
Refactor: Check for and remove duplicates in AND expressions during logic transformation.
Implement getVersion and isVersioned methods for GrailsDomainClassPersistentEntity
Ensure batch start is triggered only when events are available after waiting for sequence barrier.
Fix: Signers must have CA bit set (skip Conscrypt)
Ignore open stream analysis in main methods

Ensure I18nLanguage comparison includes external flag
Fix testNonExistentHost to use invalid hostname
Handle potential null error stream in SegmentHTTPApi
Refactor: Remove unnecessary public modifier from OnExpressionCondition
Refactor: Log dropped errors at ERROR level instead of bubbling them.
Fix: Ensure display name is set correctly in Description constructor
Ensure HAL response contains correct data type for attribute exploration

Close selector and server on shutdown
ODocument: Ensure fields are passed in pairs

Simplify select option handling in BrowserTest
Store parsed server version number for efficiency
Improve NodeCmd output clarity by separating gossip status from generation number.
Refine shader path display in property handler
Fix: Use beacon object instead of concatenated MAC address for connection.
Fix header height calculation for all-day events
Add toString() method for PostmarkResponse

Refine hetatom lookup by full atom name
Handle IOException in sslMode=PREFER case

Improve profiler logging with configuration details
Improve MapFlatSelectiveStreamReader by building value block before returning.

Add sorting to JobInstanceFetchRequest
Fix potential exception handling in JerseyServerSentEvents
Refactor: Rename addItem() to reAddItem() and add warning comment
Add checkClosed() before accessing cache entries in LockFreeCache
Improve OperationRequestCompleter by returning buffer length for specific cases.
Improve download peer selection logic in PeerGroup
Refactor: Remove unnecessary last voted for reset in FollowerState

Use async for normalize operation in watch method.
Fix whitespace issue in RollbackTest
Add refresh method test
Handle notices with Throwable instead of Exception in KafkaSupervisor

Clarify TargetType enum descriptions.

Update JDBC driver version to 1.3.5
Replace Throwables.propagate with RuntimeException for better error handling.
Return all unsent notifications (including retry queue) when shutting down
Add missing status() and type() methods to ResponseWrapper
Add option to disable JPA annotations generation

Add support for PUT requests
Safely rollback transactions in ObjectifyImpl

Use a single CountDownLatch for job execution in JobManagerTest.


Add logging for skipped type matching in NewJavaFileWriter
Disable new bytecode generator
Update ALLUXIO_DOCS_URL based on version format
Add support for closing additional closeables in Lifecycle#trying()
Refactor JPerfmonParamsDialog metric radio initialization
Improve documentation for PdfBoxGraphics2D.create()
Add missing initialization for _attachments map in UnsavedRevision
Fix: Language label font size in SpellAndGrammarCheckDialog
Add test for path creation with empty string

Set GET method for saved request in AbstractProcessingFilterTests
Fix: Correct method signature for equals and hashCode checks in ImmatureClass detection.
Simplify Emitter thread interruption handling.
Add support for observable and interest management in EmptyInventory
Remove deprecated code from getScrollListWindowHeight method
Use Deque instead of Stack for ImportStack
Clarify disableDefaults parameter description in WebSecurityConfigurerAdapter constructor.
Remove unused printCounters method
Add method for creating temporary files
Remove SQLException from insert and delete methods
Use section symbol in toString and toPatternString methods
Handle role loading errors with InternalServerErrorException
Increase maximum processing time for job service integration tests
Add onload attribute transformation for archival URL replay parsing
Sort subunit stoichiometries in descending order and update formula string accordingly.

Remove JAVA_OPTS from environment when launching Spring Boot CLI
Fix potential index underflow in RoundRobinExecuteWithFailover
Add JsonCreator for GrokPattern
Skip parsing if meta data is missing from header
Support loopback address via -b argument without XML changes (WFLY-248)
Add more informative error message for NodeUtil.getPropName()
Use Interceptor.Priority.APPLICATION for CustomPathResolver priority
Use internal network by default for TermdBuildDriver (configurable)
Add @SuppressWarnings("unchecked") annotation to fromJSON method in MessageUtils.java
Update Exec class to handle PositionedException instead of EvaluationException
Handle missing topics in KafkaSupervisor.
Only inject LiveInjector if Eclipse needs patching.
Allow empty configuration string for JChannel instantiation
Add warning if no blockchain is attached to PeerGroup
Improve UFS attribute retrieval in WebInterfaceGeneralServlet
Fix: Use correct element locator in CorrectEventFiringTest
Remove unused logger instance

Remove assertion and handle loading inner classes from InMemoryJarFile

Fix DnsCacheFile directory creation logic.
Remove unused post effect shader manager in cullAndRender method.
Use ContextFactory to create scripting context
Add mapping for JDBC type DECIMAL
Remove unused stopUFS method

Skip getExpectedLogBelief computation and return -10000 instead.
Remove unnecessary printStackTrace() call in DiscoverArchiveManifestFilesRuleProvider.
Improve PlacesSampleActivity: Add marker details and title
Fix: Correct order of parameters in batchSynonyms method
Add hashCode and equals methods to Resource class
Use ImmutableList instead of List for ALL_STATS and ALL_PAIRED_STATS
Introduce NOT_A_CHAR constant for handling invalid Unicode characters in CSV data.
Renamed TAG_H_POSITIONING_ERROR to TAG_H_POSITIONING_ERROR for clarity.


Ensure timestamp sorting in TestFilters
Fix: GroupProperty handling of null values
Refactor: Use andThenConsume instead of consumeAfter for timeout handling

Remove redundant test case for cache management with two databases
Remove unused Description class from MetricOptions
Use SEGMENT_FILE_LENGTH_DEFAULT for LOW_STORAGE_SPACE_THRESHOLD_DEFAULT
Fix potential NPE in AsyncTcpSocketImpl.java
Update default servlet version to 2.5
Ensure proper cleanup of multiple InternetExplorerDriver instances
Make all domain properties optional in GrailsDomainBinder
Use configured locale for gallery info retrieval
Add missing characters to valid pattern in WordListValidatorTest
Add default group ID for fraction list
Increase minimum allowed single upload size to 5MB.
Flush write context after updating metadata in OffHeapValueHolder
Fix ProductResultParser to handle UPC-E correctly
Change toString() output from "()" to "{}"
Add Envjs wait for browser execution in JasmineTestRunner
Add isEnabled(), setEnabled(), isRegisterManagement() and @Deprecated shouldRegisterManagement()
Fix: Use 't.header' instead of 't.header_' in MessageSerializer serialization.
Refactor ST_Explode: Find first geometry column instead of SRID
Fix LottieAnimationView invalidateDrawable behavior for non-Lottie drawables
Improve BlockLockManager by storing LockRecord instead of UserID in mLockIdToRecordMap
Read remaining bytes of current block in FileInStream
Only remove artifacts not connected to any build record from TemporaryBuildsCleaner.
Add shutdown of executor in FileCacheStore#stop()
Copy required properties for service chaining in TargetHandler

Remove unnecessary TODO comment and improve ImagePicker instantiation
Ensure SparkExecutor handles non-existent vertex messages correctly.
Use roleLocalService instead of rolePersistence for fetching roles
Remove logging level adjustments for HTTP requests
Simplify AlternatingController by removing redundant checkActiveController method.
Add closureGoogScopeAliases check for typed externs generation.


Improve filter matching in BluetoothLeScannerCompat by checking for empty filters.
Fix regex pattern compilation for case-insensitive matching
Refactor: Remove unnecessary public keyword from PerLineTailSplitter
Fix: Ensure MindmapsEngineServer starts before tests

Handle internal errors during authentication with failure tolerance option.
Add convenience methods for constructor calls
Refactor: Simplify retrieval of record attributes like @class and @rid
Fix: TransactionConfidence now correctly sets confidence type to DEAD when overridden by double-spend
Print logger and handler information instead of all other information
Make DefaultIndexRedirect a regular class instead of abstract
Ensure OAuth parameters are included in multipart requests for correct signature generation.
Remove unnecessary System.out.println()
Call onStart() and onResume() after finishCreate()
Fix user creation URL path
Rename setUp method to setUpChrome
Cleanup Speak Service references after place removal
Make runtimeBrowserString and setTestContext protected methods
Remove unused import for Sets
Use container's default ReleaseId if pom properties are not found.
Use final keyword for lockPostingChangeNotifications

Add definitionId to CaseInformation and update ProcessContextTest
Improved error handling in DefaultMetricService for metric expression evaluation.
Refactor: Initialize size, onColor and offColor in constructor instead of onLoad()
Use node.getHost() instead of node.getContainerIpAddress() for Elasticsearch client configuration.
Notify on preference header selection


Improve efficiency of table clearing by getting database connection directly in clearTable method.
Deprecate closeQuietly() due to misuse and potential exception swallowing.
Only remove session from ServletRequestContext if it matches the given session.

Return HDFSDataset.Builder instead of creating a new instance directly
Improve timer execution logic by checking for persistent timers and their execution status.
Use StringBuilder instead of StringBuffer for better performance
Reorder nodes in query search for improved performance
Refactor: Prefer exact method matches over similar ones in hierarchical methods
Fix: Idempotent close for discovered processors and workaround compiler bug
Refactor: Standardize flag names in DistributorFlags
Add support for bi-functional interfaces with Throwing.
Fix: Use package name for finding resources in ClassBundleService
Improve responsiveness of ScanProgressDialog by using SwingUtilities.invokeLater for progress updates.
Add overloaded loop() methods for gain and position control
Remove unused 'data' resource
Add missing license header and author information to PlanConfiguration.java
Format thread dump output with quotes around thread names
Improve error handling in ReflectionUtil.getDomainClass()
Refactor: Cleanup interval calculation using bitmask
Increase supported source version to 18
Fix bug in LSMSolver where no L1 penalty or bounds were considered.
Add retry logic for HTTP client exceptions and handle 401 authentication failures.

Simplify NamedSubResourceSettingBuilder.settings method parameter list

Add Jackson annotations for ErrorMessage
Add grace period configuration to TemporaryJobBuilder
Introduce `controlWindow` and `currentProduct` as nullable fields in TestOperaDriver.
Disable debug mode in Shippo API client
Add isSupervised method and update nfeatures for autoencoders
Fix screenshot comparison logic in NewSessionTest
Add sudo method for executing consumers under admin
Make credential secret volatile and add a comment about destroying it.

Refactor: Improve ControlFile set method for user-defined fields.



Add constructor and default values for Section class
Ensure termLength is cast to long before multiplication in calculateBufferCapacity()
Improve error messages in FileSystem class.
Increase cache clean time for testCacheExpiration and mark it as ignored due to concurrency issues.


Use LinkedHashMap for FlatProperties to preserve property order

Use safe() for package name in scanClasses()

Fix potential ArrayIndexOutOfBoundsException in Utils.concatBytes()
Deprecate `isSubscibe()` method and add TODO for removal in Smack 4.5.
Add 'Utils' and 'Records' to reserved type names list.

Fix: Handle _SUCCESS file path correctly in SAMFileMerger.
Ensure every function has at least one output parameter.
Use FlowRunnerTestUtil.startThread for running flow runner and cleanup runner instance.
Refine error handling in JSpecRunner
Avoid stack trace overhead in ObsoleteVersionException for improved performance.
Revert: Set VANILLA flag to true

Update ToString annotation to include superclass toString in output
Fix: Handle potential null pointer exception in rename operation
Add functionality for repeating previous commands with Enter key
Fix ServerEventTest.java: Use correct InetAddress for mirror host
Refactor: Flatten paragraph blocks in ParserUtils

Update SessionSummary state mapping for COMMITTED status.
Improve resampling precision by rounding result
Refactor: Move domain object population and update logic in DefaultScaffoldRequestHandler
Refactor: Improve Gradle command detection in GradleCli.java

Enable/disable 'Scan headers in all requests' based on 'Scan HTTP headers' selection

Optimize PathRetractionStrategy by adjusting barrier insertion logic.
Update default service name to 'mySample'
Deprecate HttpServletRequest and HttpServletResponse access in Context interface.
Improve clarity of user confirmation messages
Resolve app name before parsing parameters and launching activity

Improve SharedHealthCheckRegistries.getDefault() method
Use `time` instead of `jdate` in JDateTime hashCode calculation.
Improve pattern matching for Nullable and Nonnull annotations
Make CollectionSubject creation method static
Rename `toStringEquals` method to `strEquals` for better clarity.



Use Throwables for RuntimeException propagation in DefaultLogicResult
Add test case for URL with colon and encoded spaces in parameters/path (#4)
Add test for assertArrayTypeWithNullIsNullSafe
Only run rename view test on single server instances (ARANGO-XXXX)
Handle Ctrl+C interrupt gracefully in CliTableauResultView.

Use Google Volt Patches Stopwatch instead of Google Common Stopwatch
Handle script element parsing errors and increment use counter for errors found.

Add ResourceAccessor and initialize it in execute method.
Simplify ResourceConfigApplier interface by removing unnecessary final keyword.
Use StandardCharsets for UTF-8 encoding consistency.
Fix: Ensure Postconditions.checkReference always throws an exception
Add attribute list to BodyCells model
Add monitoring for template and asset folders
Change log level to debug
Fix: Ensure soDirectory is writable before extracting DSOs\n\nPreviously, there was no guarantee that the directory where DSOs were extracted was writable. This commit ensures that the directory is set to writable by the current process before attempting to extract any DSOs. This prevents potential IOExceptions and improves robustness.
Fix: Update active transport count logic in PublicationImage
Fix: Ensure proper room leaving even if sendStanza() throws.
Fix bug in FastAdapter.java where subItems were not correctly counted when adding items
Improve logging for pinlist synchronization.

Use Arrays.hashCode for RawType.hashCode()
Fix: Use correct `onClose` method reference in CloseableAdapter
Use ConcurrentHashMap for persist requests and jobs
Handle potential RuntimeExceptions during filter registration

Improve documentation for language model and word2vecmodel options.
Handle potential surrogate characters in StringSerializer
Refactor: Handle IllegalStateException more explicitly in DatastoreImpl
Use Number instead of Double for count retrieval

Fix: Prevent adding resources to locale groups if locked by another user.
Remove unused ScriptObjectMirror dependency
Fix: Handle potential NullPointerException in printStackTrace()
Add flush to Elasticsearch after importing VCF data\n\nThis addresses an issue where large datasets would result in "All shards failed" errors. Flushing Elasticsearch after import seems to resolve this.\n
Disable disk threshold allocation in Elasticsearch container for E2E tests
Close ByteSelectiveStreamReader resources properly
Improve logging of middleware names
Remove unused PropertiesEscapeLevel enum
Add waitAllForSafeState after adding hz3 instance in FrozenPartitionTableTest
Use time as nonce by default for synchronization across processes (#754)
Use parent class loader for extended parser classes
Prevent repeated shutdown requests

Make SurfaceLayerGL's surface field final and check for null before accessing it.
Remove unnecessary time logging from FramerManager.
Handle null response headers correctly
Fix ECKey ASN.1 parsing for stricter type checking
Improve handling of security constraints roles.
Refactor: Evaluate property invariants during lambda execution
Fix: Handle potential FileSystemNotFoundException gracefully
Allow preExecute listener to modify mojo configuration before super.configureComponent()
Prevent sending error response when document is null.
Refactor: Remove unnecessary whitespace trimming logic
Add test for setting and retrieving int1 value in ExampleBean
Refactor: Simplify branch template update logic and copy replacement function
Improve resource handling in Delombok.java
Ensure RepeatTestFilter is reset after each repeat test execution.

Use Chronology for range calculation in Period.months()
Allow `optimize` parameter to default to true in RegisteredLookupExtractionFn.
Change ServerRemoveHandler to execute at runtime stage.
Enforce consistent share value length in SecretSharing.
Add methods for splitting CamelCase strings and converting them to human-readable phrases.

Copy constructor for KeyChainGroup now creates a new ArrayList instead of directly using the input list.
Improve LegacyAddress constructor documentation
Fix: Handle Java Lang Throwable as NSException in JUnitTestRunner
Add AuthoritiesConstants import
Fix: Handle null LocalVariableInfo in isDeclaration()

Refactor: Handle FileDatastore check in isFileFound method
Refactor HelpFlags to use forEach instead of iterating with Set
Add getToolStrip() method to FeatureAttributeWindow
Run pre-crawling plugins before initial URL navigation.
Fix OutputFileMixin to handle missing parent directories
Add methods to retrieve first header and its value


Fix: Initialize titleBarBackground only when needed
Improve error handling in NatRegMrCall by providing more context in IOException.
Renamed `activity_type` property to `type` in Activity class.
Make CampAST.emit() final and return toString() for now
Add tests for ORDER BY optimization with replicated tables in outer joins and subqueries.
Add sequence number field to CommonLogFields interface.
Allow URLs starting with 'https:' in BoxBrowser
Make CanvasTestBase abstract

Fix: Ensure case-insensitive field name comparison for validators.

Fix: Clarify input/output stream direction in Streams.java
Add bootstrapGoodies() and beans() calls in Setup.bootstrap()
Use static hook for onCreate method.

Refactor FindAllQuery to handle result set iteration more efficiently
Improve class loading by using loadClass() for findClass()
Improve logging for IconEnricher
Add -jmx option to LargeState test

Fix: Change error code for StaleElementReferenceException from 13 to 10
Refactor: Use checkIfClassIsModel and isClassModel instead of direct class checks.
Fix calculation of cookie expiration time for maxAge.
Make JAXBUtil a singleton
Refine request counting logic in BrowserMobProxyTest
Fix textPane width binding and remove unnecessary logging
Add @SuppressWarnings("unchecked") annotations to test methods in ClassContextTest.
Rename TransactionRuleBuilder from AuthorityRuleBuilder

Update DiameterStackMultiplexerAS7 interface with @Local annotation.
Fix: Generate Terminate Token if not provided
Add phenotype validator and individual creation retries to Engine builder.

Refactor KafkaClient: Refresh configuration on server or group ID modification
LazyPatternBy: Implement toString and synchronize nested By retrieval
Update app logo URL to use href attribute
Fix: Use correct version in query request
Use absolute value for generated ID in SearchResultServlet
Ignore testAnimation as it is broken and not used.
Fix: Return empty list for properties in ListItems
Refactor: Change how MutableBySettingFieldOfField sets indirect fields
Update javax.* imports to jakarta.*
Only generate HGVS code for SNVs
Use system property 'm2.repo' for Maven repository directory
Optimize PathMatchingSimpleStorageResourcePatternResolver by removing redundant existence check.
Add flag to track receipt of first fragment task message in JoinProducer
Add since tag for ModelObjectWithChildren
Avoid unnecessary gap activation when delay is not generated.

Fix: Ensure foreign key references are correctly added to changelog graph
Add validation for surrogate pairs in DatabaseUtils.escapeString()
Fix: Skip initialization in edit mode for RippleView
Refactor: Simplify getContent method and remove duplicate code
Add tree processing callback for TSMeta
Fix: Use RepositoryService instead of RuntimeService for creating deployments
Add test for none rows with other threads and exact queries

Fix List constructor logic for handling element type.
Simplify message handling completion logic in Puller.
Improve formatting of low storage warning messages
Fix: Use click() instead of setSelected() for radio buttons
Update GRADLE_VERSIONS list with Gradle 6.2
Add SPARQL service description namespace and update properties accordingly.
Use long literals for calculations in getMaxMemory() method.
Improve LoadDataChange performance by optimizing prepared statement usage.
Add support for TRANSIENT modifier in AdapterUtil
Fix: Pass null for defaultLabel in MetricParams creation
Use natural order of string representation for ordering

Reduce thread pool size in TestMultiThread for better performance.
Handle remote link close and detach events.
Disable testCreateBucketedTable due to timeout issues.
Improve bitmap decoding concurrency
Rename parameter '_predicate' to '_a' for consistency.

Fix handling of null columns in Index toString()


Fix SoundManager drain() issue causing thread hangs
Fix: Throw more informative exception when creating empty JavaFileObjects
Fix: LocalDate.ofYearDay test range
Refactor: Improve UserProfile constructor clarity and deprecate old constructor
Refactor: Simplify loading of EvolutionResult from stream
Ignore StageTest due to ongoing refactoring.

Change type of `flags` field from short to byte
Update OptimalCECPParameters with generics type change.
Fix rendering mode serialization in ScheduleRenderer
Synchronize `myLooper()` method in ShadowLooper.
Add separator and filename to AbstractFile toString()
Add check for empty string before appending HTML to report content.
Ensure uniqueness of flag encoders.
Fix: Convert file separators to forward slashes in site generation
Cache deserialized value in ClientDelegatingFuture.
Add support for SelenideElement in Describe#toString()

Use CPU time instead of scheduled time for plan statistics
Make CloudSpannerOAuthUtil non-instantiable
Improve CREATE TABLE SQL statement for MySQL writer
Add methods to get URI from Path and read file contents as stream.
Use weak hash map for SlaveReconnectionWork to avoid leaking Slaves.

Refactor: Move WeaselProgram to io.jenetics.ext package
Fix: Correctly extract missing resource names from property fields.
Use constant-time comparison for salt and hash equality in HashData object.
Refactor: Only map nodes with type 'node' in DocumentGraphMapper
Use JCommander's parse method instead of creating an unused instance



Fix reading class file header on JDK7 and later

Improve DagManager's hasRunningJobs method for clarity and accuracy.
Remove shutdown hook for graph database
Use string id for LinkedIn identity
WorkerOptions: Set default workers count to 1
Fix typo in Java exclusion pattern example

Add transaction support information for outbound resource adapters.
Fix CulpritsRecipientProvider handling of non-AbstractBuild runs
Fix AudioDisconnectEvent firing when not connected and close audio connection on disconnect
Use flipped buffer for data in InstallRequest
Add Preconditions check in JoinMatcher accept method
Fix: Adjust content extraction range in SourceCode
Fix Windows support for large ChronicleMaps\n\nPreviously, a warning was issued on Windows when creating a ChronicleMap with a large size. This commit now throws an IllegalStateException if the map size exceeds Windows limitations, preventing potential issues.
Improve partition filtering in HiveMetadata#getTableLayout()

Add commit() method call to configure FileDownloader
Improve RefreshListener logic for handling events targeting other applications.
Update exception handling in ScheduledReporter to catch all exceptions.
Fix: Calculate endColumn correctly when there's an error on a new line.
Refactor: Use dedicated method for ID element check
Refine private identifier detection to exclude double underscores.
Make WebSocketSubscriptionTransport constructor public
Make ViewPropertyDescriptor.newInstance() return null by default
Change error message in FileList activity to 'cancel' instead of 'error'
Fix: Initialize adapter only if it's not already set.
Change log level for interrupted sleep in SQL batch execution from ERROR to DEBUG.
Add constructor and Jackson annotations for HostIdentifier
Add check for escape character before escaping extra characters in CSVWriter.
Make Structure field non-transient
Improved parsing of GPSD data by attempting double conversion for non-time fields.
Simplify and canonicalize automata during normalization process.
Add default artifact type for SVN connector configuration

Use UNDEFINED instead of null for undefined type variables.

Add support for messenger extensions and fallback URLs in WebButton

Refactor: Change log level from DEBUG to TRACE for DiscoveryEntryStoreInMemory.
Use Spring Specifications Executor for counting entities
Remove clear() method from StandardTitanGraph
Use channelUri directly in ingress publication creation

Add Javadoc comment to testCovtypeGBM method
Add name property to GeoJsonObject
Introduce conductor cycle threshold configuration.
Handle potential RejectedExecutionException during resource cleanup

Remove task registration from Task.run() method.
Fix: Work around compiler bug in closing processor class loader
Handle missing displayName from GitHub API response
Add readForRecycle method to ByteBuf
Fix null check for comment field in upload method
Fix: Propagate error when it's not on subscribe
Update RepackageMojo to use artifact extension for output file name
Close DBCursor after iterating through GridFSDBFile results.

Add constructor for Meta with no assigned id
Improve sigmoid function and gradient calculation for ObjFunction
Update SslTest to use TLS_DHE_RSA_WITH_AES_256_GCM_SHA384 for MariaDB server.
Fix spelling correction for "unswar" and "d zwar"
Update DataIO.unpackInt() method parameter name from 'is' to 'in'

Improve thread handling in PlannerBenchmarkConfig. Use more threads for 'AUTO' setting.
Add setters for slow and very slow invocation thresholds with @ManagedAttribute annotation.
Start stats agent after construction and remove redundant OpsAgent registration.
Refactor: Use PathTools.getPathsOfLengthUpto for improved efficiency
Add check for manual provider before fetching documentation
Handle file syncing for both zip and non-zip files
Refactor: Rename `getAllExtensions` to `getExtensions` and update usage
Update email intent with string resources
Allow specifying a single entity when creating a HibernateBundle
Add handling for x-amz-acl header in PutObject requests
Refactor: Simplify scroll notification handling
Add scope context for XbaseTypeComputer
Fix: Use Var() for index operand in ForAll loops
Register plugins after loading the webview.
Add version information to Transaction toString()
Remove debug logging from showMenuItem chain
Remove debug logging from ExceptionProxy
Only forward declare private local types in ObjectiveCImplementationGenerator.
Warm up partitions in SmallClusterTest
Use boolean for SINGLETON option
Use getPreviousSuccessfulBuild() instead of getPreviousNotFailedBuild() for dependency changes

Use scheduleWithFixedDelay instead of scheduleAtFixedRate for latency marking timer.
Reduce logging verbosity for SelectorOptimizer.
Simplify exception handling in TestUtil
Add alt text to type image for Selenium discrimination
Add node role information to disk usage stats
Use resolved types for type arguments in AbstractBeanDefinitionBuilder
Add `--provision` flag when checking Vagrant state as RUNNING

Add check for field size equality when comparing Avro schemas

Fix: Ensure brand files are not empty before processing.
Fix potential NPE when closing PythonPlanStreamer server.
Remove unused usage count tracking from ClassDescriptor
Add toString() method to FluxMaterialize

Log exception when creating AgiScript instances
Update author name in ProductRegisterEndPoint
Refactor: Simplify PUBREC constructor
Add test for large date format in PactDslJsonBody
Remove unnecessary System.err.println in CollapsePropertiesTest.
Use Msc.isStandalone() to determine production mode

Update GrpcServerComponent to include appSecurity-5.0 in security check.
Allow for both 'Etc/UTC' and 'GMT' as identifiers for UTC time zone.
Reduce `NUM_FILES_PER_CHUNK` for improved load leveling and performance.
Fix: Conditional logic for adding corpora to translation request body
Allow testNotifyArgument for protocol version 2 on PostgreSQL 9.0 and above
Fix: Remove expired listeners from rejected list by index
Add logging for FeatureRequirement assumptions
Enable StrictMode for debugging
Update SnapshotRestore to use consistent JSON key names.
Add toString method and handle empty location in ArtifactQuery
Clarify DiskLruCache documentation regarding editing entries
Update thread pool size in AsyncScalr

Fix handling of incomplete chunks in CaddScoreParser



Improve JSON Schema Validation Test by adding repositories with different schemas
Fix LazilyLoadedCtor to handle cases where class constructor is null
Update candidate term ID in cluster mark file
Only create analysis summary if findbugs.noSummary is not set
Remove userId field from OrganizationDto
Refactor: Improve OperaDriver navigation logic and idle waiting
Refactor: Improve clarity and consistency in `toDate` method.
Simplify Script execution handling
Refine PartiallyConstructedObjectAccess detection by considering method visibility.
Update copyright year in AnalyzerPhase.java
Fix corrupted hint name parsing in HintedHandOffManager

Add resource path to model (#SONAR-4245)
Increase open files limit to 1024
Improve backpressure handling by adding more granular checks and delivery logic.
Refactor: Remove redundant type validation for long properties in ShardingProperties.
Fix: Use Context instead of Fragment in SettingService creation
Fix: Assert thread acquired locks in PidRuntimeFactoryTest
Refactor: Remove redundant journal flush in rename operation
Add access check cache to Fake class for Permit annotation
Fix potential buffer overflow in truncated entry writing.\n\nThe previous implementation of writeTruncatedEntry() didn't account for the overhead of the object header when writing truncated entries. This could lead to a buffer overflow if the truncated data used up more space than the original compressed buffer. \n\nThis commit fixes the issue by adding the OBJECT_HEADER_BYTES to the compressedLength before writing the truncated entry.
Improve cancellation handling in SqlStageExecution

Improve ExchangeOperator efficiency by closing exchange client when operator is done or all clients are complete.
Change TreeNodeLongClickListener interface to return boolean
Fix: Ensure ValidationFailed has at least one problem.
Improve temp file handling in ServerIntegrationTestBase
Refactor: Remove unnecessary code for subresource setting
Add PriorityResource binding to SingularityResourceModule
Fix: Correctly update ThreadLocal values for response object in FunctionContext.
Fix bug in XMLRuleHandler: Ensure end position correction doesn't exceed token count.
Increase benchmark slice size for better performance
Fix: Add support for DB2 in primary key validation.
Fix term id handling in Publication for interrupt scenarios
Add support for specifying a static constructor name in @Data annotation
Prevent snapshots during bootstrap.
Only add tools.jar to classloader when compiling.
Only log debug messages if enabled

Refactor: Use 'shouldNotBeNull' instead of Assert.assertNotNull()
Reduce refresh calls in PeriodicalRefreshStrategyTest
Refactor: Move environment setting into execute method
Update SearchUI with user information and font loading
Fix: Adjust initial store index for method receiver.
Improve KieControllerSwaggerIntegrationTest test by accepting both 'Controller Documentation' and 'Business Central Documentation' titles.
Fix calculation of safe mode exit time in DefaultSafeModeManager
Add test for encoding with special characters
Refactor: Improve Timeslot Duration Validation in SseBuilder
Increase timeout for NettyTransportTest methods
Fix: Access token check in MPRestClient
Use String for attribute names in CmdLineAnnotatorUtils
Improve Task.isRunning() logic for better accuracy.
Remove redundant waitForClusterToForm calls in MultipleCacheManagersTest
Handle empty Type parameter list in BeanSpec#getType
Fix: Ensure performance numbers are not returned for non-existent or empty cases.
Refactor: Separate layer state from shared state
Fix: Continue type resolution even with unknown/error types in lists and fields.
Add getters for PolicyRule and PolicyRuleURL
Add X-Text-Delivered header to LogResource.
Add support for onRequestPermissionsResult callback.
Use native method for interrupted() call

Use config.getMaxTextLengthAnonymous() for max text length API endpoint
Deprecate all Map methods as ContentMap is immutable
Fix: Prevent RuntimeException when normalizing URIs with newlines or tabs
Ensure whitespace trimming within CDATA sections only when wiki syntax is present.
